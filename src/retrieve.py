"""检索:doc_ids 锁定范围 + BM25 多 query 粗筛 + rerank 精排。

设计权衡(受 500 万 token 预算约束):
- 放弃对整份 300 页财报做全量 embedding(实测每题 ~40 万 token,爆预算)。
- 保留研究证明性价比最高的 cross-encoder rerank(PwC:MRR +59%)。
- BM25(多 query)先粗筛出 pool 个候选,再用 qwen3-reranker 精排到 top_k。
- rerank 候选正文截断到 600 字,控制 rerank 的 token 开销。
"""
from __future__ import annotations

import re

import jieba
from rank_bm25 import BM25Okapi

from parse import build_doc

_STOP = set("的 了 和 与 及 或 在 是 为 对 以 等 中 上 下 之 其 该 本 第 条 款 项 者 "
            "。 ， 、 ； ： （ ） 《 》 “ ” ？ ! ? . , ; :".split())


def _tokenize(text: str) -> list[str]:
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and t not in _STOP]


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion:融合多个 query 的 BM25 排序。"""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return scores


class DocRetriever:
    """针对单题引用的若干文档构建 BM25 索引,配合 rerank 精排。"""

    def __init__(self, domain: str, doc_ids: list[str], llm=None) -> None:
        self.llm = llm
        self.doc_ids = doc_ids
        self.chunks: list[dict] = []
        for did in doc_ids:
            self.chunks.extend(build_doc(domain, did))
        self._corpus_tokens = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self._corpus_tokens) if self.chunks else None

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
        # 题干常用中文引号标出真正的检索主题（如“施救费用”、
        # “犹豫期”）。精确短语的区分度远高于“产品/条款/正确”等通用词。
        anchor_phrases = {
            x.strip()
            for x in re.findall(r"[“\"]([^”\"]{2,24})[”\"]", opt_text)
            if x.strip()
        }
        generic_ngrams = {
            "以下说法", "下列说法", "以下哪些", "正确的是",
            "文档内容", "条款规定", "以下描述", "相关财务",
        }
        # 四字连续片段比 jieba 单词更能保留指标语义，例如
        # “研发投入占营业收入比例”。
        char_ngrams: set[str] = set()
        for span in re.findall(r"[一-鿿]{4,}", opt_text):
            char_ngrams.update(span[i:i + 4] for i in range(len(span) - 3))
        char_ngrams -= generic_ngrams
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
            # 长词元和题干引号短语作为主题锚点。这能把表格中一个
            # 数字巧合命中的片段，排在真正讨论目标条款的片段之后。
            s += sum(1.5 for term in opt_terms if len(term) >= 4 and term in txt)
            s += sum(5.0 for phrase in anchor_phrases if phrase in txt)
            s += min(6.0, sum(0.35 for gram in char_ngrams if gram in txt))
            # 表格优先
            if table_boost and c.get("is_table"):
                s += 2.0
            return s

        # 保留一部分 BM25/RRF 原始名次先验，防止“年份数字巧合”
        # 将真正命中完整指标短语的第 1-2 名候选挤出。
        prior = {idx: pos for pos, idx in enumerate(cand)}
        return sorted(
            cand,
            key=lambda i: score(i) + 4.0 / (1.0 + prior[i] * 0.35),
            reverse=True,
        )

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
            cand = self._local_rerank(cand, options, domain)[:per_doc]
            balanced.extend(cand)

        # 2) 统一 rerank 精排(可选);保证不丢均衡覆盖
        if self.llm is not None and len(balanced) > top_k:
            docs = [self.chunks[i]["text"][:600] for i in balanced]
            order = self.llm.rerank(query, docs, top_n=len(balanced))
            if order:
                balanced = [balanced[o] for o in order]

        return [self.chunks[i] for i in balanced[: max(top_k, per_doc * n_docs)]]

    def retrieve_for_option(
        self,
        queries: list[str],
        option_text: str,
        *,
        pool: int = 40,
        top_k: int = 5,
        domain: str = "",
        use_rerank: bool = False,
        prefer_doc_balance: bool = False,
        allowed_doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """为单个选项检索证据，保留跨文档覆盖。

        与 :meth:`retrieve` 的区别是这里不把四个选项混在一起打分。调用方可以
        传入 Agent 规划阶段生成的多个事实查询，每个文档先独立召回，再统一精排。
        这让最终上下文天然形成 ``option -> evidence[]`` 的证据矩阵。
        """
        if not self.bm25:
            return []

        allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
        active_doc_ids = [
            doc_id for doc_id in self.doc_ids
            if allowed is None or doc_id in allowed
        ]
        if not active_doc_ids:
            return []

        def allowed_rank(query: str) -> list[int]:
            return [
                idx for idx in self._bm25_rank(query)
                if self.chunks[idx]["doc_id"] in active_doc_ids
            ]

        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            clean_queries = [option_text]

        n_docs = len(active_doc_ids)
        # 每份文档先产生一个保底候选，其余名额由所有文档按选项相关度竞争。
        # 这比平均配额更适合“多个引用文档、但某个选项只对应其中一份”的题目。
        per_doc_depth = max(2, (top_k + n_docs - 1) // n_docs)
        per_doc_candidates: list[list[int]] = []

        for did in active_doc_ids:
            doc_idx = [i for i, c in enumerate(self.chunks) if c["doc_id"] == did]
            if not doc_idx:
                continue
            idx_set = set(doc_idx)
            rank_lists = []
            for query in clean_queries:
                ranks = [i for i in self._bm25_rank(query) if i in idx_set][:pool]
                rank_lists.append(ranks)
            fused = _rrf(rank_lists)
            cand = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]
            reranked = self._local_rerank(cand, [option_text], domain)
            # 调用方把最精确的指标/条款查询放在首位。保留其 BM25
            # 第一名，再用多特征重排补充，避免数字巧合将它挤出。
            anchor = rank_lists[0][0] if rank_lists and rank_lists[0] else None
            cand = ([anchor] if anchor is not None else []) + [
                i for i in reranked if i != anchor
            ]
            cand = cand[:per_doc_depth]
            per_doc_candidates.append(cand)

        # 真正的跨文档比较需要每份材料都有足够证据；而一个
        # 选项只对应一份产品条款时，仍保留“每文档1块+全局竞争”。
        keep_per_doc = max(1, top_k // n_docs) if prefer_doc_balance else 1
        balanced: list[int] = [
            idx
            for cand in per_doc_candidates
            for idx in cand[:keep_per_doc]
        ][:top_k]

        # 剩余位置按所有查询的全局 RRF + 选项数字/关键词重排竞争。
        global_rank_lists = [allowed_rank(q)[:pool] for q in clean_queries]
        global_scores = _rrf(global_rank_lists)
        global_ranked = sorted(global_scores, key=lambda i: global_scores[i], reverse=True)
        seen = set(balanced)
        global_ranked = self._local_rerank(
            [i for i in global_ranked if i not in seen], [option_text], domain
        )
        balanced.extend(global_ranked[: max(0, top_k - len(balanced))])

        # 极端情况下某份文档没有候选，再按全局结果补齐。
        seen = set(balanced)
        if len(balanced) < top_k:
            all_ranks = [_rrf([allowed_rank(q)[:pool]]) for q in clean_queries]
            global_scores: dict[int, float] = {}
            for scores in all_ranks:
                for idx, score in scores.items():
                    global_scores[idx] = global_scores.get(idx, 0.0) + score
            rest = sorted(global_scores, key=lambda i: global_scores[i], reverse=True)
            rest = self._local_rerank(
                [i for i in rest if i not in seen], [option_text], domain
            )
            balanced.extend(rest[: top_k - len(balanced)])

        # 可选的选项级 cross-encoder。精排只改变顺序，不丢掉文档均衡候选。
        if use_rerank and self.llm is not None and len(balanced) > 1:
            docs = [self.chunks[i]["text"][:700] for i in balanced]
            rerank_query = "\n".join(clean_queries[:4])
            order = self.llm.rerank(rerank_query, docs, top_n=len(balanced))
            if order:
                balanced = [balanced[i] for i in order]

        return [self.chunks[i] for i in balanced[:top_k]]


def build_context(chunks: list[dict], max_chars: int = 10000) -> str:
    """把召回 chunk 拼成上下文,带来源标注,控制总长度。"""
    parts = []
    used = 0
    for c in chunks:
        header = f"【来源: {c['doc_id']} #chunk{c['chunk_id']}】\n"
        body = c["text"]
        if used + len(header) + len(body) > max_chars:
            body = body[: max(0, max_chars - used - len(header))]
        parts.append(header + body)
        used += len(header) + len(body)
        if used >= max_chars:
            break
    return "\n\n".join(parts)
