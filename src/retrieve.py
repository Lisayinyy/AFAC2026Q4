"""检索:doc_ids 锁定范围 + BM25 多 query 粗筛 + rerank 精排。

设计权衡(受 500 万 token 预算约束):
- 放弃对整份 300 页财报做全量 embedding(实测每题 ~40 万 token,爆预算)。
- 保留研究证明性价比最高的 cross-encoder rerank(PwC:MRR +59%)。
- BM25(多 query)先粗筛出 pool 个候选,再用 qwen3-reranker 精排到 top_k。
- rerank 候选正文截断到 600 字,控制 rerank 的 token 开销。
"""
from __future__ import annotations

import re

try:
    import jieba
except ImportError:  # 离线诊断仍可使用确定性分词降级
    jieba = None
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:
    _BM25Okapi = None

from parse import build_doc
from segment.financial_report import (expand_financial_groups,
                                      segment_financial_report_chunks)
from segment.insurance import expand_parent_groups, segment_insurance_chunks
from segment.regulatory import (expand_regulatory_groups,
                                segment_regulatory_chunks)

_STOP = set("的 了 和 与 及 或 在 是 为 对 以 等 中 上 下 之 其 该 本 第 条 款 项 者 "
            "。 ， 、 ； ： （ ） 《 》 “ ” ？ ! ? . , ; :".split())
_CN_DIGITS = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
    "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
    "十": "10",
}

_INSURANCE_PRECISE_TERMS = {
    "宽限期", "效力中止", "效力恢复", "等待期", "犹豫期", "免赔额",
    "特定药品", "院外", "保单贷款", "现金价值", "施救费用", "处方审核",
}

_FINANCIAL_METRICS = (
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "归属于上市公司股东的净利润",
    "经营活动产生的现金流量净额",
    "研发投入占营业收入比例",
    "资本化研发投入占研发投入的比例",
    "现金及现金等价物净增加额",
    "加权平均净资产收益率",
    "营业总收入", "营业收入", "研发投入金额", "研发费用",
    "基本每股收益", "稀释每股收益", "总资产", "净资产",
    "现金分红",
)


def _financial_metric_anchors(text: str) -> list[str]:
    normalized = (
        text.replace("的比例", "比例")
        .replace("经营活动现金流净额", "经营活动产生的现金流量净额")
        .replace("经营活动现金流量净额", "经营活动产生的现金流量净额")
        .replace("归母净利润", "归属于上市公司股东的净利润")
        .replace("营业总收入", "营业收入")
        .replace("营收", "营业收入")
    )
    return [metric for metric in _FINANCIAL_METRICS if metric in normalized]


def _insurance_intent_types(text: str) -> set[str]:
    intents: set[str] = set()
    if re.search(r"效力中止|效力恢复|合同效力|合同终止|合同解除|复效", text):
        intents.add("contract_state")
    if "宽限期" in text:
        intents.add("grace_period")
    if "等待期" in text:
        intents.add("waiting_period")
    if re.search(r"免赔额|计算方法|计算公式|给付比例", text):
        intents.add("deductible_formula")
    if re.search(r"责任免除|免责|不承担|不涵盖|不在保障范围", text):
        intents.add("exclusion")
    if re.search(r"保险责任|保险金|保障|涵盖|费用", text):
        intents.add("coverage")
    return intents


def normalize_numeric_text(text: str) -> str:
    """统一千分位、数字空格与常见中文百分比表达。"""
    text = re.sub(r"(?<=\d)[,，](?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[万亿亿元元%])", "", text)

    def replace_percent(match: re.Match) -> str:
        value = _CN_DIGITS.get(match.group(1))
        return f"{value}%" if value is not None else match.group(0)

    return re.sub(r"百分之([零〇一二两三四五六七八九十])", replace_percent, text)


def _tokenize(text: str) -> list[str]:
    text = normalize_numeric_text(text)
    tokens = (
        jieba.lcut(text) if jieba is not None
        else re.findall(r"[A-Za-z0-9.%]+|[\u4e00-\u9fff]{2}", text)
    )
    return [t.strip() for t in tokens if t.strip() and t not in _STOP]


def _query_variants(query: str) -> list[str]:
    normalized = normalize_numeric_text(query)
    return [query] if normalized == query else [query, normalized]


class _FallbackBM25:
    def __init__(self, corpus: list[list[str]]) -> None:
        import math
        self.corpus = corpus
        self.avgdl = sum(map(len, corpus)) / max(1, len(corpus))
        df: dict[str, int] = {}
        for row in corpus:
            for term in set(row):
                df[term] = df.get(term, 0) + 1
        count = len(corpus)
        self.idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for row in self.corpus:
            counts: dict[str, int] = {}
            for term in row:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            for term in set(query):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                score += self.idf.get(term, 0.0) * (tf * 2.2) / (
                    tf + 1.2 * (0.7 + 0.3 * len(row) / max(1, self.avgdl))
                )
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
    """相关性与多样性联合选块，避免同一章节的重叠块挤满证据。"""
    remaining = list(dict.fromkeys(candidates))
    selected: list[int] = []
    region_counts: dict[tuple, int] = {}
    while remaining and len(selected) < limit:
        best = None
        best_value = float("-inf")
        for index in remaining:
            chunk = chunks[index]
            key = (chunk.get("doc_id"), chunk.get("region", chunk.get("chunk_id")))
            if region_counts.get(key, 0) >= region_cap:
                continue
            redundancy = max(
                (
                    _term_overlap(chunk["text"], chunks[other]["text"])
                    for other in selected
                    # The same metric from two annual reports is required evidence,
                    # not duplication. Penalize overlap only inside one document.
                    if chunks[other].get("doc_id") == chunk.get("doc_id")
                ),
                default=0.0,
            )
            value = scores.get(index, 0.0) - 0.35 * redundancy
            if chunk.get("is_table"):
                value += 0.08
            if value > best_value:
                best, best_value = index, value
        if best is None:
            break
        remaining.remove(best)
        selected.append(best)
        chunk = chunks[best]
        key = (chunk.get("doc_id"), chunk.get("region", chunk.get("chunk_id")))
        region_counts[key] = region_counts.get(key, 0) + 1
    return selected


class DocRetriever:
    """针对单题引用的若干文档构建 BM25 索引,配合 rerank 精排。"""

    def __init__(self, domain: str, doc_ids: list[str], llm=None) -> None:
        self.llm = llm
        self.domain = domain
        self.doc_ids = doc_ids
        # ``source_chunks`` powers document identity/full-text statistics. ``chunks``
        # is the retrieval index and may contain much smaller domain-specific atoms.
        self.source_chunks: list[dict] = []
        for did in doc_ids:
            self.source_chunks.extend(build_doc(domain, did))
        if domain == "insurance":
            self.chunks = segment_insurance_chunks(self.source_chunks)
        elif domain == "financial_reports":
            self.chunks = segment_financial_report_chunks(self.source_chunks)
        elif domain == "regulatory":
            self.chunks = segment_regulatory_chunks(self.source_chunks)
        else:
            self.chunks = list(self.source_chunks)
        self._corpus_tokens = [
            _tokenize(c.get("search_text") or c["text"]) for c in self.chunks
        ]
        engine = _BM25Okapi or _FallbackBM25
        self.bm25 = engine(self._corpus_tokens) if self.chunks else None

    def _reasoning_evidence(self, indexes: list[int], limit: int) -> list[dict]:
        records = [self.chunks[index] for index in indexes]
        if self.domain == "insurance":
            return expand_parent_groups(records, limit=limit)
        if self.domain == "financial_reports":
            return expand_financial_groups(records, limit=limit)
        if self.domain == "regulatory":
            return expand_regulatory_groups(records, limit=limit)
        return records[:limit]

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
        opt_text = normalize_numeric_text(" ".join(options))
        opt_nums = set(re.findall(r"\d[\d\.%]*", opt_text))
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
        insurance_intents = _insurance_intent_types(opt_text) if domain == "insurance" else set()
        metric_anchors = (
            _financial_metric_anchors(opt_text)
            if domain == "financial_reports" else []
        )
        regulatory_role = ""
        if domain == "regulatory":
            if re.search(r"申辩|辩称|听证意见", opt_text):
                regulatory_role = "party_argument"
            elif re.search(r"监管认定|我会认为|本局认为|处罚决定", opt_text):
                regulatory_role = "authority_finding"
            elif re.search(r"案例事实|违法事实|经查明", opt_text):
                regulatory_role = "case_fact"
            else:
                regulatory_role = "formal_rule"

        def score(i: int) -> float:
            c = self.chunks[i]
            txt = normalize_numeric_text(c.get("search_text") or c["text"])
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
            if insurance_intents and c.get("fact_type") in insurance_intents:
                s += 5.0
            if metric_anchors and c.get("fact_type") == "financial_metric":
                metric_name = str(c.get("qualifiers", {}).get("metric", ""))
                matched_metric = max(
                    (anchor for anchor in metric_anchors if anchor in metric_name),
                    key=len, default="",
                )
                if matched_metric:
                    s += 12.0 + min(4.0, len(matched_metric) / 6)
                    if re.search(r"主要会计数据|财务指标", c.get("parent_title", "")):
                        s += 3.0
                else:
                    # A number appearing in an unrelated row (e.g. shareholder or
                    # subsidiary table) must not outrank the exact metric label.
                    s -= 3.0
            if regulatory_role:
                if c.get("fact_type") == regulatory_role:
                    s += 5.0
                elif regulatory_role == "formal_rule" and c.get("fact_type") == "party_argument":
                    s -= 4.0
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
        queries = []
        for raw_query in [query] + [f"{query} {o}" for o in options]:
            queries.extend(_query_variants(raw_query))

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

        result_limit = max(top_k, per_doc * n_docs)
        return self._reasoning_evidence(balanced, result_limit)

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

        clean_queries = []
        for raw_query in queries:
            if raw_query and raw_query.strip():
                clean_queries.extend(_query_variants(raw_query.strip()))
        clean_queries = list(dict.fromkeys(clean_queries))
        if not clean_queries:
            clean_queries = [option_text]
        financial_metric_anchors = (
            _financial_metric_anchors(option_text)
            if domain == "financial_reports" else []
        )

        active_doc_ids = [
            doc_id for doc_id in self.doc_ids
            if not allowed_doc_ids or doc_id in allowed_doc_ids
        ]
        if not active_doc_ids:
            active_doc_ids = self.doc_ids
        doc_candidates: dict[str, list[int]] = {}
        fallback_coverage_docs: set[str] = set()
        precise_terms = {
            term for term in _INSURANCE_PRECISE_TERMS
            if term in option_text
        } if domain == "insurance" else set()
        state_intent = bool(precise_terms & {"宽限期", "效力中止", "效力恢复"})
        negative_coverage = bool(re.search(r"不涵盖|不包括|不在保障范围", option_text))
        for doc_id in active_doc_ids:
            indexes = [
                index for index, chunk in enumerate(self.chunks)
                if chunk["doc_id"] == doc_id
            ]
            if precise_terms and not any(
                any(term in (self.chunks[index].get("search_text") or self.chunks[index]["text"])
                    for term in precise_terms)
                for index in indexes
            ):
                # An absent, highly specific term is meaningful for insurance. For
                # a closed-list coverage negation, retrieve the product's positive
                # responsibility list; for contract-state questions, return no
                # borrowed clause at all.
                if negative_coverage and not state_intent:
                    indexes = [
                        index for index in indexes
                        if self.chunks[index].get("fact_type") == "coverage"
                    ]
                    fallback_coverage_docs.add(doc_id)
                else:
                    indexes = []
            doc_candidates[doc_id] = indexes
        allowed_indexes = {
            index for indexes in doc_candidates.values() for index in indexes
        }
        if not allowed_indexes:
            return []
        n_docs = max(1, len(active_doc_ids))
        # 每份文档先产生一个保底候选，其余名额由所有文档按选项相关度竞争。
        # 这比平均配额更适合“多个引用文档、但某个选项只对应其中一份”的题目。
        per_doc_depth = max(2, (top_k + n_docs - 1) // n_docs)
        per_doc_candidates: list[list[int]] = []

        for did in active_doc_ids:
            doc_idx = doc_candidates.get(did, [])
            if not doc_idx:
                continue
            idx_set = set(doc_idx)
            rank_lists = []
            doc_queries = (
                ["保险责任 保障项目 保险金 给付"]
                if did in fallback_coverage_docs else clean_queries
            )
            for query in doc_queries:
                ranks = [i for i in self._bm25_rank(query) if i in idx_set][:pool]
                rank_lists.append(ranks)
            fused = _rrf(rank_lists)
            cand = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]
            if financial_metric_anchors:
                structured = [
                    index for index in doc_idx
                    if self.chunks[index].get("fact_type") == "financial_metric"
                    and any(
                        anchor in str(
                            self.chunks[index].get("qualifiers", {}).get("metric", "")
                        )
                        for anchor in financial_metric_anchors
                    )
                ]
                cand = list(dict.fromkeys(structured + cand))
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
        global_rank_lists = [
            [index for index in self._bm25_rank(query) if index in allowed_indexes][:pool]
            for query in clean_queries
        ]
        global_scores = _rrf(global_rank_lists)
        global_ranked = sorted(global_scores, key=lambda i: global_scores[i], reverse=True)
        if financial_metric_anchors:
            structured_global = [
                index for index in allowed_indexes
                if self.chunks[index].get("fact_type") == "financial_metric"
                and any(
                    anchor in str(
                        self.chunks[index].get("qualifiers", {}).get("metric", "")
                    )
                    for anchor in financial_metric_anchors
                )
            ]
            global_ranked = list(dict.fromkeys(structured_global + global_ranked))
        seen = set(balanced)
        global_ranked = self._local_rerank(
            [i for i in global_ranked if i not in seen], [option_text], domain
        )
        balanced.extend(global_ranked[: max(0, top_k - len(balanced))])

        # 极端情况下某份文档没有候选，再按全局结果补齐。
        seen = set(balanced)
        if len(balanced) < top_k:
            all_ranks = [
                _rrf([[
                    index for index in self._bm25_rank(query)
                    if index in allowed_indexes
                ][:pool]])
                for query in clean_queries
            ]
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

        # 从更宽的候选池做区域感知 MMR。重叠滑窗在词面上高度相似，若不去重，
        # 选项的第二个关键事实或另一张表格很容易被挤出最终证据矩阵。
        candidate_pool = list(dict.fromkeys(
            balanced + global_ranked[: max(top_k * 3, top_k)]
        ))
        relevance = {
            index: 1.0 / (position + 1)
            for position, index in enumerate(candidate_pool)
        }
        selected = _mmr_select(
            candidate_pool, relevance, self.chunks, top_k,
            region_cap=2 if prefer_doc_balance else 1,
        )
        if prefer_doc_balance:
            for doc_id in active_doc_ids:
                if any(self.chunks[i]["doc_id"] == doc_id for i in selected):
                    continue
                replacement = next(
                    (i for i in candidate_pool if self.chunks[i]["doc_id"] == doc_id),
                    None,
                )
                if replacement is not None:
                    if len(selected) >= top_k:
                        selected[-1] = replacement
                    else:
                        selected.append(replacement)
        evidence = self._reasoning_evidence(selected, top_k)
        if fallback_coverage_docs:
            for item in evidence:
                if item["doc_id"] in fallback_coverage_docs:
                    item["retrieval_mode"] = "closed_coverage_list"
            # A few representative responsibility groups are enough to establish
            # the product's closed list; sending ten near-duplicate benefits wastes
            # context without strengthening the absence inference.
            if evidence and all(
                item.get("retrieval_mode") == "closed_coverage_list"
                for item in evidence
            ):
                evidence = evidence[:5]
        return evidence


def build_context(chunks: list[dict], max_chars: int = 10000) -> str:
    """拼接上下文，并只在完整句子或完整表格行边界截断。"""
    parts = []
    used = 0
    for c in chunks:
        section = c.get("section") or "未标注章节"
        header = (
            f"【来源: {c['doc_id']} #chunk{c['chunk_id']} "
            f"章节:{section} 区域:{c.get('region', '?')}】\n"
        )
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        body = c["text"]
        if c.get("is_table"):
            units = [line for line in body.splitlines() if line.strip()]
            joiner = "\n"
        else:
            units = [
                sentence.strip()
                for sentence in re.findall(r"[^。！？!?；;]+[。！？!?；;]?", body)
                if sentence.strip()
            ]
            joiner = ""
        kept: list[str] = []
        size = 0
        for unit in units:
            extra = len(unit) + (len(joiner) if kept else 0)
            if size + extra > remaining:
                break
            kept.append(unit)
            size += extra
        body = joiner.join(kept)
        if not body:
            continue
        parts.append(header + body)
        used += len(header) + len(body)
        if used >= max_chars:
            break
    return "\n\n".join(parts)
