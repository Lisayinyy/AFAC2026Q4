"""检索:doc_ids 锁定范围 + BM25 多 query 粗筛 + rerank 精排。

设计权衡(受 500 万 token 预算约束):
- 放弃对整份 300 页财报做全量 embedding(实测每题 ~40 万 token,爆预算)。
- 保留研究证明性价比最高的 cross-encoder rerank(PwC:MRR +59%)。
- BM25(多 query)先粗筛出 pool 个候选,再用 qwen3-reranker 精排到 top_k。
- rerank 候选正文截断到 600 字,控制 rerank 的 token 开销。
"""
from __future__ import annotations

import re

try:  # 可选依赖：离线评估和无模型运行不应被阻塞
    import jieba
except ImportError:  # pragma: no cover - 仅在精简环境触发
    jieba = None
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:  # pragma: no cover
    _BM25Okapi = None

from parse import build_doc

_STOP = set("的 了 和 与 及 或 在 是 为 对 以 等 中 上 下 之 其 该 本 第 条 款 项 者 "
            "。 ， 、 ； ： （ ） 《 》 “ ” ？ ! ? . , ; :".split())


def _tokenize(text: str) -> list[str]:
    if jieba is not None:
        tokens = jieba.lcut(text)
    else:
        # 中文二元词 + 数字/英文词，保证金融指标和公司名不被整句吞掉。
        tokens = re.findall(r"[A-Za-z0-9.%]+|[\u4e00-\u9fff]{2}", text)
    return [t.strip() for t in tokens if t.strip() and t not in _STOP]


class _FallbackBM25:
    """无 rank_bm25 时的确定性 BM25-lite，参数与标准实现同量纲。"""
    def __init__(self, corpus: list[list[str]]) -> None:
        import math
        self.corpus = corpus
        self.avgdl = sum(map(len, corpus)) / max(1, len(corpus))
        df: dict[str, int] = {}
        for row in corpus:
            for term in set(row):
                df[term] = df.get(term, 0) + 1
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        q = set(query)
        for row in self.corpus:
            counts: dict[str, int] = {}
            for term in row:
                counts[term] = counts.get(term, 0) + 1
            dl = len(row)
            score = 0.0
            for term in q:
                if term not in counts:
                    continue
                tf = counts[term]
                score += self.idf.get(term, 0.0) * (tf * 2.2) / (tf + 1.2 * (0.7 + 0.3 * dl / max(1, self.avgdl)))
            scores.append(score)
        return scores


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion:融合多个 query 的 BM25 排序。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return scores


def _term_overlap(left: str, right: str) -> float:
    a, b = set(_tokenize(left)), set(_tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _mmr_select(candidates: list[int], scores: dict[int, float], chunks: list[dict],
                limit: int, region_cap: int = 2) -> list[int]:
    """按相关性+多样性选块，限制同一区域重复挤占上下文。"""
    selected: list[int] = []
    region_counts: dict[tuple, int] = {}
    while candidates and len(selected) < limit:
        best = None
        best_score = float("-inf")
        for i in candidates:
            c = chunks[i]
            key = (c.get("doc_id"), c.get("region", c.get("chunk_id")))
            if region_counts.get(key, 0) >= region_cap:
                continue
            redundancy = max(
                (_term_overlap(c["text"], chunks[j]["text"]) for j in selected),
                default=0.0,
            )
            value = scores.get(i, 0.0) - 0.35 * redundancy
            # 表格/标题块在平分时略优先，但不突破文档和区域配额。
            if c.get("is_table"):
                value += 0.08
            if value > best_score:
                best, best_score = i, value
        if best is None:
            break
        candidates.remove(best)
        selected.append(best)
        c = chunks[best]
        key = (c.get("doc_id"), c.get("region", c.get("chunk_id")))
        region_counts[key] = region_counts.get(key, 0) + 1
    return selected


def _ensure_doc_coverage(candidates: list[int], selected: list[int],
                         scores: dict[int, float], chunks: list[dict],
                         doc_ids: list[str], limit: int) -> list[int]:
    """在 MMR 后补齐每个引用文档的最佳证据，不突破 top-k。"""
    selected_set = set(selected)
    for doc_id in doc_ids:
        if any(chunks[i].get("doc_id") == doc_id for i in selected):
            continue
        choices = [i for i in candidates if i not in selected_set and chunks[i].get("doc_id") == doc_id]
        if not choices:
            continue
        best = max(choices, key=lambda i: scores.get(i, 0.0))
        if len(selected) >= limit:
            # 替换当前最低分且同文档已有多个块的项，避免丢掉其它文档。
            replaceable = [i for i in selected if sum(chunks[j].get("doc_id") == chunks[i].get("doc_id") for j in selected) > 1]
            if not replaceable:
                continue
            worst = min(replaceable, key=lambda i: scores.get(i, 0.0))
            selected[selected.index(worst)] = best
            selected_set.remove(worst)
        else:
            selected.append(best)
        selected_set.add(best)
    return selected[:limit]


class DocRetriever:
    """针对单题引用的若干文档构建 BM25 索引,配合 rerank 精排。"""

    def __init__(self, domain: str, doc_ids: list[str], llm=None) -> None:
        self.llm = llm
        self.doc_ids = doc_ids
        self.chunks: list[dict] = []
        for did in doc_ids:
            self.chunks.extend(build_doc(domain, did))
        self._corpus_tokens = [_tokenize(c["text"]) for c in self.chunks]
        engine = _BM25Okapi or _FallbackBM25
        self.bm25 = engine(self._corpus_tokens) if self.chunks else None

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)

    def _local_rerank(self, cand: list[int], options: list[str],
                      domain: str) -> list[int]:
        """本地零成本重排:选项词/数字重叠 + 表格优先(财报/保险)。

        比纯 BM25 更贴题:优先带有选项里出现的数字/关键词的 chunk;
        财报/保险数值题优先表格块。避免额外 LLM 调用(受限流约束)。
        """
        # 选项里的数字与较长词元
        opt_text = " ".join(options)
        opt_nums = set(re.findall(r"\d[\d,\.%]*", opt_text))
        opt_terms = set(t for t in _tokenize(opt_text) if len(t) >= 2)
        table_boost = domain in ("financial_reports", "insurance")

        def score(i: int) -> float:
            c = self.chunks[i]
            txt = c["text"]
            s = 0.0
            # 数字重叠(数值题关键)
            for n in opt_nums:
                if len(n) >= 2 and n in txt:
                    s += 3.0
            # 词元重叠
            toks = set(_tokenize(txt))
            s += len(opt_terms & toks) * 0.5
            # 表格优先
            if table_boost and c.get("is_table"):
                s += 2.0
            return s

        # 稳定排序:先按本地分,同分保持 BM25 原序
        return sorted(cand, key=lambda i: score(i), reverse=True)

    def retrieve(self, query: str, options: list[str],
                 pool: int = 20, top_k: int = 8, domain: str = "") -> list[dict]:
        """按 doc 配额均衡召回,避免多文档题里某个文档被挤占(漏选主因)。

        对每个引用文档单独跑 BM25 多 query + 本地重排,各取 top_k/n_docs;
        再对合并候选用 qwen3-rerank 统一精排(不足则保留均衡结果)。
        """
        if not self.bm25:
            return []
        queries = [query] + [f"{query} {o}" for o in options]

        # 1) 按 doc 均衡:每个文档独立取候选
        n_docs = max(1, len(self.doc_ids))
        per_doc = max(4, top_k // n_docs + 2)  # 每文档配额,略放宽
        balanced: list[int] = []
        for did in self.doc_ids:
            doc_idx = [i for i, c in enumerate(self.chunks) if c["doc_id"] == did]
            if not doc_idx:
                continue
            idx_set = set(doc_idx)
            rank_lists = []
            for q in queries:
                ranked = [i for i in self._bm25_rank(q) if i in idx_set][:pool]
                rank_lists.append(ranked)
            fused = _rrf(rank_lists)
            cand = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]
            cand = self._local_rerank(cand, options, domain)
            balanced.extend(_mmr_select(cand, fused, self.chunks, per_doc))

        # 2) 统一 rerank 精排(可选);保证不丢均衡覆盖
        if self.llm is not None and len(balanced) > top_k:
            docs = [self.chunks[i]["text"][:600] for i in balanced]
            order = self.llm.rerank(query, docs, top_n=len(balanced))
            if order:
                balanced = [balanced[o] for o in order]

        # 最终再做一次跨文档 MMR，保留每份文档的证据覆盖，并过滤重复 overlap 块。
        fused_final = {i: 1.0 / (pos + 1) for pos, i in enumerate(balanced)}
        selected = _mmr_select(
            list(dict.fromkeys(balanced)), fused_final, self.chunks,
            top_k, region_cap=1,
        )
        selected = _ensure_doc_coverage(
            list(dict.fromkeys(balanced)), selected, fused_final,
            self.chunks, self.doc_ids, top_k,
        )
        return [self.chunks[i] for i in selected]

    def retrieve_option_evidence(self, question: str, option: str,
                                 *, top_k: int = 3, domain: str = "") -> list[dict]:
        """单个选项的窄召回。

        共享召回解决跨文档关系，窄召回解决某个选项的数字/否定词被其它
        选项挤掉的问题。由于复用同一个 BM25 索引，不增加模型 Token。
        """
        return self.retrieve(
            f"{question} {option}", [option], pool=24,
            top_k=top_k, domain=domain,
        )


def build_context(chunks: list[dict], max_chars: int = 10000) -> str:
    """把召回 chunk 拼成上下文,带来源标注,控制总长度。"""
    parts = []
    used = 0
    for c in chunks:
        section = c.get("section") or "未标注章节"
        header = (f"【来源: {c['doc_id']} #chunk{c['chunk_id']} "
                  f"章节:{section} 区域:{c.get('region', '?')}】\n")
        body = c["text"]
        if used + len(header) + len(body) > max_chars:
            body = body[: max(0, max_chars - used - len(header))]
        parts.append(header + body)
        used += len(header) + len(body)
        if used >= max_chars:
            break
    return "\n\n".join(parts)
