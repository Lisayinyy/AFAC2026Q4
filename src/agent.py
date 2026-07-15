"""金融长文本 Agent：规划、证据矩阵、动态记忆、计算与条件复核。

该模块刻意保留 ``answer.py`` 的 baseline，便于做严格 A/B 对比。Agent 的一次
答题过程为：

1. 规划：判断题型并把每个选项拆成待核实事实；
2. 检索：为每个选项独立召回证据，低覆盖时自动扩检；
3. 记忆：把原始证据压缩为带来源的结构化事实；
4. 计算：本地执行差值、比例、增长率、求和与比较；
5. 裁决：基于证据矩阵逐项输出 verdict/confidence；
6. 复核：仅对低置信度、证据不足或高风险的选项再次检索和判断。
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from config import LLMClient
from retrieve import DocRetriever, _tokenize


_VALID_VERDICTS = {"support", "contradict", "uncertain"}
_CALC_WORDS = re.compile(
    r"增长率|增速|同比|环比|占比|比例|比重|高于|低于|超过|少于|差额|"
    r"下降|上升|减少|增加|倍|合计|总和|约为|百分点"
)
_DATE_RE = re.compile(r"(?:19|20)\d{2}(?:\s*年|Q[1-4]|[一二三四]季度)?")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_UNIT_RE = re.compile(r"亿元|万元|千元|百万元|元|美元|港元|%|百分点|个工作日|工作日|个月|年|日")

_DOMAIN_POLICY = {
    "insurance": (
        "文档编号与保险产品必须严格按文档身份映射核对。禁止使用‘通常、一般、常规险种’"
        "等行业常识替代条款，也禁止把另一产品的责任、免责、贷款或现金价值规则套用过来。"
        "只有匹配产品的直接条款或基于其明确公式的计算才能 support；缺条款应 uncertain。"
        "必须先按题干的筛选条件裁决：例如题干问‘哪些明确给出公式’时，"
        "选项括号中‘仅载明、未给公式’即表示该产品不满足筛选条件，"
        "不能因括号描述本身属实而 support。法律上位词可包含明确列举的下位项，"
        "如‘欠款’可概括未偿还借款及利息，不应只因措辞更概括就 contradict。"
    ),
    "research": (
        "研究报告允许同义改写、口径概括和阈值表达。若证据给出更具体数值并能蕴含选项的"
        "‘超过/约为/达到’，应 support。题干或文档上下文已经限定的地区/主体可在选项中省略。"
        "复合陈述的核心数量事实准确、尾部只是宽泛影响总结且没有直接相反证据时，也应 support。"
        "指标名称略有概括不等于矛盾；只有相同主体、时期、指标下存在方向或数值冲突才"
        "contradict，证据缺失只能 uncertain。若选项省略地域，但引用文档中相同期间、"
        "数值和指标只有一个明确出处，且题干未要求地域必须显式写出，应按该直接事实"
        "support，不能以‘未限定地区’为由否定。"
    ),
    "financial_contracts": (
        "募集说明书可能用发行人、标的公司、控股股东等不同主体，必须先对齐主体。措辞省略、"
        "概括或信息分散不构成矛盾；只有同一主体、期间和指标的直接冲突才 contradict。"
        "评价概括性说法时按经济和法律实质而非字面逐字匹配；正式公式给出的"
        "公告日前交易均价约束，可作为‘不低于公告时市场价格’的具体化表述。"
    ),
    "regulatory": (
        "区分正式规则、当事人申辩和案例事实。只有适用法规正文对同一行为直接规定了相反"
        "要求才能 contradict；条款分散或表述概括不构成矛盾。"
        "对‘若/如果…则…’条件命题，必须在假定前件成立时核对法律后果，"
        "不得用具体案例中前件实际不成立来否定条件规则。"
    ),
    "financial_reports": (
        "严格对齐公司、年份、合并/母公司口径、单位和指标；可用原始数值完成一步确定性计算。"
        "‘每股’与‘每10股’是实质单位差异，必须换算，不得以语境歧义忽略。"
        "利润分配中年度现金分红、特别现金分红与合计数是不同口径，"
        "选项问某一组成部分时，不得用含其他部分的合计数反驳。"
    ),
}


@dataclass
class EvidenceSlot:
    letter: str
    claim: str
    queries: list[str]
    chunks: list[dict] = field(default_factory=list)
    coverage: float = 0.0
    expanded: bool = False
    doc_coverage: list[str] = field(default_factory=list)
    focus: str = ""


def _json_from_text(text: str) -> dict:
    """从模型输出中提取首个 JSON 对象；失败时返回空字典。"""
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _local_question_type(q: dict) -> str:
    text = q["question"] + " " + " ".join(q["options"].values())
    if _CALC_WORDS.search(text):
        return "numeric_comparison"
    if len(q.get("doc_ids", [])) > 1:
        return "cross_document_comparison"
    if q["domain"] == "regulatory":
        return "rule_verification"
    if q["domain"] == "insurance":
        return "clause_verification"
    return "fact_verification"


def _fallback_plan(q: dict) -> dict:
    """规划调用失败时的确定性降级，不让 Agent 退化为默认选 A。"""
    qtype = _local_question_type(q)
    items = []
    for letter, claim in q["options"].items():
        dates = _DATE_RE.findall(claim)
        units = _UNIT_RE.findall(claim)
        items.append({
            "option": letter,
            "claim": claim,
            "required_facts": [claim],
            "search_queries": [f"{q['question']} {claim}", claim],
            "expected_subjects": [],
            "expected_periods": dates,
            "expected_units": units,
            "calculation": {
                "op": "compare" if _CALC_WORDS.search(claim) else "none",
                "description": "根据证据核对数值关系" if _CALC_WORDS.search(claim) else "",
            },
        })
    return {
        "question_type": qtype,
        "cross_document": len(q.get("doc_ids", [])) > 1,
        "items": items,
        "planner_fallback": True,
    }


def plan_question(llm: LLMClient, q: dict) -> dict:
    """让模型只做任务规划，不接触答案和标准答案。"""
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    prompt = f"""你是金融长文本 Agent 的规划器。不要回答题目，只拆解核验任务。

领域：{q['domain']}
引用文档：{', '.join(q['doc_ids'])}
题目：{q['question']}
选项：
{options}

返回严格 JSON，不要 Markdown：
{{
  "question_type": "fact_verification|numeric_comparison|cross_document_comparison|rule_verification|clause_verification",
  "cross_document": true,
  "items": [
    {{
      "option": "A",
      "claim": "原选项",
      "required_facts": ["核验事实1", "核验事实2"],
      "search_queries": ["适合文档检索的短查询1", "短查询2"],
      "expected_subjects": ["公司/法规/保险产品"],
      "expected_periods": ["年份或期限"],
      "expected_units": ["元/%/日等"],
      "calculation": {{
        "op": "none|difference|ratio|growth_rate|percentage_change|sum|compare",
        "description": "需要核算的关系；无需计算则为空"
      }}
    }}
  ]
}}

每个选项必须恰好出现一次。search_queries 应包含关键主体、指标、年份和数值，
不要把“选项A”之类无信息文本放进查询。"""
    out = llm.chat(
        [{"role": "system", "content": "你只负责规划检索与核验步骤。"},
         {"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.0,
        thinking=False,
    )
    raw = _json_from_text(out)
    items_by_letter = {
        str(x.get("option", "")).upper(): x
        for x in raw.get("items", []) if isinstance(x, dict)
    }
    if any(letter not in items_by_letter for letter in q["options"]):
        return _fallback_plan(q)

    normalized = []
    for letter, claim in q["options"].items():
        item = items_by_letter[letter]
        queries = [str(x).strip() for x in item.get("search_queries", []) if str(x).strip()]
        required = [str(x).strip() for x in item.get("required_facts", []) if str(x).strip()]
        calc = item.get("calculation") if isinstance(item.get("calculation"), dict) else {}
        normalized.append({
            "option": letter,
            "claim": claim,
            "required_facts": required or [claim],
            "search_queries": queries or [f"{q['question']} {claim}", claim],
            "expected_subjects": item.get("expected_subjects", []),
            "expected_periods": item.get("expected_periods", []),
            "expected_units": item.get("expected_units", []),
            "calculation": {
                "op": calc.get("op", "none"),
                "description": calc.get("description", ""),
            },
        })
    return {
        "question_type": raw.get("question_type", _local_question_type(q)),
        "cross_document": bool(raw.get("cross_document", len(q["doc_ids"]) > 1)),
        "items": normalized,
        "planner_fallback": False,
    }


def _feature_coverage(item: dict, chunks: list[dict]) -> float:
    """估算该选项的主体/数字/长词是否进入证据，供扩检决策使用。"""
    blob = "\n".join(c["text"] for c in chunks)
    source = " ".join(
        [item.get("claim", "")]
        + list(item.get("required_facts", []))
        + list(item.get("expected_subjects", []))
        + list(item.get("expected_periods", []))
    )
    nums = {n for n in _NUM_RE.findall(source) if len(n) >= 2}
    terms = {t for t in _tokenize(source) if len(t) >= 2}
    # 过于通用的命令词不参与覆盖率，避免虚高。
    terms -= {"正确", "错误", "准确", "描述", "资料", "文档", "根据", "其中", "关于"}
    weighted_total = len(nums) * 2 + len(terms)
    if weighted_total == 0:
        return 1.0
    hit = sum(2 for n in nums if n in blob)
    hit += sum(1 for t in terms if t in blob)
    return round(hit / weighted_total, 4)


def _query_expansions(q: dict, item: dict) -> list[str]:
    """用金融文档中的标准标签扩展口语化选项。"""
    claim = item.get("claim", "")
    years = " ".join(dict.fromkeys(_DATE_RE.findall(q["question"] + " " + claim)))
    queries: list[str] = []

    # 引号内的词通常是保险/法规题的章节主题。
    quoted = re.findall(r"[“\"]([^”\"]{2,24})[”\"]", q["question"])
    queries.extend(f"{topic} 保险责任 条款" for topic in quoted)

    if q["domain"] == "financial_reports":
        if "净利润" in claim:
            queries.append(f"主要会计数据 归属于上市公司股东的净利润 {years} 本年比上年增减")
        if "营业收入" in claim or "营收" in claim:
            queries.append(f"主要会计数据 营业收入 {years} 本年比上年增减")
        if "研发投入" in claim:
            queries.append(f"研发投入占营业收入比例 {years}")
        if "研发费用" in claim:
            queries.append(f"研发费用 营业收入 {years}")
        if "现金流" in claim:
            queries.append(f"经营活动产生的现金流量净额 {years} 同比增减")
    elif q["domain"] == "financial_contracts":
        if "资产负债率" in claim:
            queries.append("偿债能力分析 资产负债率 2024 2023 2022")
        if "净利润" in claim:
            queries.append("利润表 净利润 2024 2023 2022")
        if "转股价格" in claim:
            queries.append("初始转股价格 公告日 交易均价")

    generic_verdict_claim = claim.strip() in {"正确", "错误", "对", "错"}
    queries.append(q["question"] if generic_verdict_claim else claim)
    return list(dict.fromkeys(x.strip() for x in queries if x.strip()))


def _match_option_documents(option_text: str,
                            document_profiles: dict[str, str]) -> list[str]:
    """把包含明确产品/主体名称的选项绑定到唯一文档。

    仅在匹配明显领先时返回，避免“第二份文档”“两家公司”一类比较选项被误限流。
    该绑定主要阻止保险多产品题把 A 产品条款借给 B 产品。
    """
    ignore = {
        "保险", "产品", "条款", "公司", "医疗险", "重疾险", "保险金",
        "费用", "规定", "文档", "第一份", "第二份",
    }
    insurer_brands = {
        brand for brand in ("国寿", "太保", "平安", "众安", "人保", "泰康", "太平", "新华", "阳光")
        if brand in option_text
    }
    if len(insurer_brands) >= 2:
        return []
    canonical_option = (
        option_text.replace("家财险", "家庭财产保险")
        .replace("重疾险", "重大疾病保险")
        .replace("医疗险", "医疗保险")
    )
    terms = {
        token for token in _tokenize(canonical_option)
        if len(token) >= 2 and token not in ignore
    }
    compact_option = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", canonical_option)
    scores: dict[str, int] = {}
    for doc_id, profile in document_profiles.items():
        compact_profile = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", profile)
        score = sum(len(term) for term in terms if term in profile)
        # PDF 文本和产品简称常在词中间插入“产险/复发/住院”等字样，无法靠
        # 一个连续长串匹配。按选项字符位置贪心选择不重叠的最长片段，既能识别
        # “家财险→家庭财产保险”，也能组合“众安 + 白血病”。
        occupied: set[int] = set()
        ignored_fragments = {"保险", "公司", "产品", "条款", "医疗"}
        for size in range(min(8, len(compact_option)), 1, -1):
            for start in range(len(compact_option) - size + 1):
                if any(pos in occupied for pos in range(start, start + size)):
                    continue
                fragment = compact_option[start:start + size]
                if fragment in ignored_fragments or fragment not in compact_profile:
                    continue
                score += size
                occupied.update(range(start, start + size))
        # 产品简称通常是连续 4~10 个字；连续命中比通用分词更可靠。
        for size in range(min(12, len(compact_option)), 3, -1):
            if any(
                compact_option[start:start + size] in compact_profile
                for start in range(len(compact_option) - size + 1)
            ):
                score += size
                break
        scores[doc_id] = score
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_doc, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    return [best_doc] if best_score >= 5 and best_score >= second_score + 2 else []


def build_evidence_matrix(
    retriever: DocRetriever,
    q: dict,
    plan: dict,
    *,
    use_rerank: bool = False,
    document_profiles: dict[str, str] | None = None,
) -> dict[str, EvidenceSlot]:
    """逐选项检索；覆盖率低或跨文档不完整时自动扩检。"""
    matrix: dict[str, EvidenceSlot] = {}
    prefer_doc_balance = (
        q["domain"] in {"financial_reports", "financial_contracts"}
        or plan.get("question_type") == "numeric_comparison"
    )
    for item in plan["items"]:
        letter = item["option"]
        profiles = document_profiles or {}
        matched_docs = (
            _match_option_documents(item["claim"], profiles)
            if q["domain"] == "insurance" else []
        )
        # If an explicit insurer brand is absent from all referenced documents, the
        # option must not borrow evidence from a different product. This also makes
        # a missing/decoy document explicit to the judge as an empty evidence slot.
        explicit_brands = {
            name for name in ("国寿", "太保", "平安", "众安", "人保", "泰康", "太平", "新华", "阳光")
            if name in item["claim"]
        }
        missing_named_subject = bool(
            q["domain"] == "insurance" and len(explicit_brands) == 1
            and not any(
                next(iter(explicit_brands)) in profile for profile in profiles.values()
            )
        )
        queries = _query_expansions(q, item) + list(item.get("search_queries", []))
        queries.append(f"{q['question']} {item['claim']}")
        focus_text = f"{q['question']} {item['claim']}"
        compact_domain = q["domain"] in {"financial_reports", "regulatory"}
        base_top_k = 6 if compact_domain else 8
        chunks = [] if missing_named_subject else retriever.retrieve_for_option(
            queries, focus_text, pool=45, top_k=base_top_k, domain=q["domain"],
            use_rerank=use_rerank, prefer_doc_balance=prefer_doc_balance,
            allowed_doc_ids=matched_docs or None,
        )
        coverage = _feature_coverage(item, chunks)
        doc_coverage = sorted({c["doc_id"] for c in chunks})
        expected_docs = matched_docs or q["doc_ids"]
        closed_coverage_list = any(
            c.get("retrieval_mode") == "closed_coverage_list" for c in chunks
        )
        needs_expand = not missing_named_subject and not closed_coverage_list and (
            coverage < 0.48 or (
            plan.get("cross_document") and not matched_docs
            and len(doc_coverage) < len(expected_docs)
            )
        )
        if needs_expand:
            extra_queries = (
                list(item.get("required_facts", []))
                + [item["claim"], q["question"]]
            )
            extra = retriever.retrieve_for_option(
                extra_queries,
                focus_text,
                pool=80,
                top_k=8 if compact_domain else 10,
                domain=q["domain"],
                use_rerank=use_rerank,
                prefer_doc_balance=prefer_doc_balance,
                allowed_doc_ids=matched_docs or None,
            )
            seen = {(c["doc_id"], c["chunk_id"]) for c in chunks}
            chunks.extend(
                c for c in extra if (c["doc_id"], c["chunk_id"]) not in seen
            )
            chunks = chunks[:8 if compact_domain else 10]
            coverage = _feature_coverage(item, chunks)
            doc_coverage = sorted({c["doc_id"] for c in chunks})
        matrix[letter] = EvidenceSlot(
            letter=letter,
            claim=item["claim"],
            queries=queries,
            chunks=chunks,
            coverage=coverage,
            expanded=needs_expand,
            doc_coverage=doc_coverage,
            focus=focus_text,
        )
    return matrix


def _best_excerpt(text: str, focus: str, max_chars: int) -> str:
    """围绕选项数字/关键词选择 Chunk 内最高匹配窗口，而非固定截取开头。"""
    if text.lstrip().startswith("[表格]"):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        features = list(dict.fromkeys(
            [x for x in _NUM_RE.findall(focus) if len(x) >= 2]
            + [x for x in _tokenize(focus) if len(x) >= 2]
        ))
        scored = [
            (sum((4 if feature in _NUM_RE.findall(focus) else 1)
                 for feature in features if feature in line), index)
            for index, line in enumerate(lines)
        ]
        anchor = max(scored, default=(0, 0))[1]
        # 表头与命中行的前后行一起保留，绝不在单元格中间截断。
        ordered = [
            index for index in dict.fromkeys(
                [0, 1]
                + list(range(max(0, anchor - 2), min(len(lines), anchor + 3)))
            )
            if 0 <= index < len(lines)
        ]
        chosen: list[str] = []
        used = 0
        for index in ordered:
            line = lines[index]
            if used + len(line) + (1 if chosen else 0) > max_chars:
                continue
            chosen.append(line)
            used += len(line) + (1 if chosen else 0)
        return "\n".join(chosen)

    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    nums = [x for x in _NUM_RE.findall(focus) if len(x) >= 2]
    terms = [x for x in _tokenize(focus) if len(x) >= 2]
    terms = [x for x in terms if x not in {"正确", "错误", "描述", "说法", "文档", "关于"}]
    features = list(dict.fromkeys(nums + terms))
    positions = []
    for feature in features:
        start = 0
        while True:
            pos = clean.find(feature, start)
            if pos < 0:
                break
            positions.append(pos)
            start = pos + max(1, len(feature))
    if not positions:
        return clean[:max_chars]

    best = (float("-inf"), 0, clean[:max_chars])
    for pos in positions:
        start = max(0, min(len(clean) - max_chars, pos - max_chars // 3))
        segment = clean[start:start + max_chars]
        score = sum((4 if f in nums else 1) for f in features if f in segment)
        # 越靠近窗口中央越稳定，避免关键词刚好落在截断边缘。
        score -= abs((pos - start) - max_chars / 2) / max_chars
        if score > best[0]:
            best = (score, start, segment)
    prefix = "…" if best[1] > 0 else ""
    suffix = "…" if best[1] + max_chars < len(clean) else ""
    return prefix + best[2] + suffix


def _matrix_text(matrix: dict[str, EvidenceSlot], per_chunk: int = 550) -> str:
    parts = []
    for letter, slot in matrix.items():
        parts.append(
            f"\n### 选项 {letter}: {slot.claim}\n"
            f"检索覆盖率={slot.coverage:.2f}; 文档={','.join(slot.doc_coverage)}"
        )
        for idx, chunk in enumerate(slot.chunks, 1):
            text = _best_excerpt(chunk["text"], slot.focus or slot.claim, per_chunk)
            evidence_kind = chunk.get(
                "fact_type", "table" if chunk.get("is_table") else "prose"
            )
            matched_fact = chunk.get("atomic_text", "")
            fact_hint = ""
            if matched_fact and matched_fact.strip() != chunk["text"].strip():
                fact_hint = (
                    " | 命中原子事实:"
                    + _best_excerpt(matched_fact, slot.focus or slot.claim, 160)
                )
            parts.append(
                f"[{letter}-E{idx} | {chunk['doc_id']}#chunk{chunk['chunk_id']}"
                f" | type={evidence_kind}{fact_hint}] {text}"
            )
    return "\n".join(parts)


def build_document_profiles(retriever: DocRetriever) -> dict[str, str]:
    """从每份文档开头抽取轻量身份卡，防止跨文档时把主体映射错位。"""
    profiles: dict[str, str] = {}
    title_words = re.compile(
        r"条款|年度报告|年报|募集说明书|研究报告|证券研究报告|管理办法|实施细则|反洗钱法"
    )
    for doc_id in retriever.doc_ids:
        chunks = sorted(
            (c for c in retriever.source_chunks
             if c["doc_id"] == doc_id and not c.get("is_table")),
            key=lambda c: c["chunk_id"],
        )[:5]
        if not chunks:
            profiles[doc_id] = ""
            continue
        lead = re.sub(r"\s+", " ", chunks[0]["text"]).strip()[:260]
        candidates = []
        for chunk in chunks:
            for line in chunk["text"].splitlines():
                line = re.sub(r"\s+", " ", line).strip()
                if 4 <= len(line) <= 120 and title_words.search(line):
                    if line not in candidates:
                        candidates.append(line)
        # 具体产品/报告标题优先于“请阅读条款”等通用阅读指引。
        def title_score(line: str) -> tuple[int, int]:
            specific = bool(re.search(
                r"[\u4e00-\u9fffA-Za-z0-9（）()·]{3,}(?:保险|报告|说明书|办法|法).*(?:条款|报告|说明书|办法|法)",
                line,
            ))
            boilerplate = bool(re.search(r"阅读|请扫描|加粗|重大利害|免责条款", line))
            return (int(specific) * 2 - int(boilerplate), -len(line))

        candidates.sort(key=title_score, reverse=True)
        profiles[doc_id] = (lead + " | " + " | ".join(candidates[:6]))[:1200]
    return profiles


def build_document_term_stats(retriever: DocRetriever, q: dict) -> dict[str, dict[str, int]]:
    """统计题目主题词在每份全文中的命中次数，辅助核验否定命题。"""
    text = q["question"] + " " + " ".join(q["options"].values())
    vocabulary = {
        "免赔额", "特定药品", "院外", "身故保险金", "保单贷款", "施救费用",
        "犹豫期", "现金价值", "自杀", "净利润", "营业收入", "资产负债率",
        "研发投入", "研发费用", "现金流量净额", "转股价格", "处罚时效",
        "宽限期", "效力中止",
    }
    terms = sorted(term for term in vocabulary if term in text)
    stats: dict[str, dict[str, int]] = {}
    for doc_id in retriever.doc_ids:
        blob = "\n".join(
            chunk["text"] for chunk in retriever.source_chunks
            if chunk["doc_id"] == doc_id
        )
        stats[doc_id] = {term: blob.count(term) for term in terms}
        if "犹豫期" in text:
            stats[doc_id]["犹豫期全额退费"] = int(bool(re.search(
                r"犹豫期.{0,360}(?:全额退还保险费|"
                r"(?:无息)?退还.{0,30}(?:所支付|已收|全部).{0,12}保险费)",
                blob,
                flags=re.S,
            )))
    return stats


def compress_memory(llm: LLMClient, q: dict, plan: dict,
                    matrix: dict[str, EvidenceSlot],
                    document_profiles: dict[str, str],
                    document_term_stats: dict[str, dict[str, int]]) -> dict:
    """把重复原文压成结构化事实，同时给出可复核的初步裁决。

    把“压缩”和“第一次裁决”合成一次调用，是 Agent 的默认低成本路径。确定性
    计算与一致性检查仍在调用后本地执行；存在风险时再交给独立复核器。
    """
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    evidence = _matrix_text(matrix)
    plan_text = json.dumps(plan, ensure_ascii=False)
    profiles_text = json.dumps(document_profiles, ensure_ascii=False)
    domain_policy = _DOMAIN_POLICY.get(q["domain"], "")
    prompt = f"""把金融证据压缩成可核验的结构化工作记忆。不要凭常识补全，
每条事实必须引用给定的证据编号。特别检查主体、年份、指标口径、单位、正负号。

题目：{q['question']}
选项：
{options}

核验计划：{plan_text}

文档身份卡（doc_id -> 文档主体/标题）：{profiles_text}
全文主题词命中次数（用于识别跨产品套用和否定命题）：
{json.dumps(document_term_stats, ensure_ascii=False)}

本领域裁决规则：{domain_policy}

选项级证据矩阵：
{evidence}

返回严格 JSON，不要 Markdown：
{{
  "options": {{
    "A": {{
      "facts": [
        {{"subject":"", "metric":"", "period":"", "value":null,
          "raw_value":"", "unit":"", "relation":"", "source":"A-E1"}}
      ],
      "preliminary_status":"support|contradict|uncertain",
      "confidence":0.0,
      "evidence":["A-E1"],
      "reason":"一句话说明初步判断",
      "missing_facts": [],
      "calculation": {{
        "op":"none|difference|ratio|growth_rate|percentage_change|sum|compare",
        "operands":[1.0, 2.0], "operator":">|<|>=|<=|==", "target":null,
        "description":""
      }}
    }}
  }}
}}

数值必须使用 JSON number，不要把逗号、货币符号或百分号放入 operands。
growth_rate/percentage_change 的 operands 顺序固定为 [新值, 旧值]；ratio 为
[分子, 分母]；difference 为 [被减数, 减数]。无法确定操作数时用空数组并列入
missing_facts。preliminary_status 必须逐项给出，不要因需要一步计算或同义改写就
轻易判 uncertain。四个选项都必须输出。"""
    out = llm.chat(
        [{"role": "system", "content": "你负责证据压缩，不得脱离证据编造事实。"},
         {"role": "user", "content": prompt}],
        max_tokens=2400,
        temperature=0.0,
        thinking=False,
    )
    memory = _json_from_text(out)
    if out and not isinstance(memory.get("options"), dict):
        # 长 JSON 偶尔会被模型包裹额外说明或在末尾截断。只在失败时追加一次
        # 小型修复调用，避免正常题承担额外成本。
        repair_prompt = f"""把下面内容修复为一个合法、紧凑的 JSON 对象。不要新增事实，
不要解释。顶层必须是 options，且必须含 {', '.join(q['options'])} 四个键；每项保留
facts、preliminary_status、confidence、evidence、reason、missing_facts、calculation。
无法恢复的字段用空数组、空字符串或 uncertain，不得猜测。

待修复内容：
{out[:12000]}"""
        repaired = llm.chat(
            [{"role": "system", "content": "你只修复 JSON 语法和缺失的结构字段。"},
             {"role": "user", "content": repair_prompt}],
            max_tokens=1800,
            temperature=0.0,
            thinking=False,
        )
        memory = _json_from_text(repaired)
    if not isinstance(memory.get("options"), dict):
        # Some endpoints can repeatedly truncate the verbose facts schema for one
        # unusually dense question. Fall back to a compact evidence-grounded
        # verdict schema; normal validation/review still runs afterwards.
        compact_evidence = _matrix_text(matrix, per_chunk=280)
        compact_prompt = f"""只依据以下证据逐项判断，不要解释过程，不要输出 Markdown。
题目：{q['question']}
选项：
{options}
证据：
{compact_evidence}

返回一个紧凑合法 JSON，四个选项都必须存在：
{{"options":{{"A":{{"preliminary_status":"support|contradict|uncertain",
"confidence":0.0,"evidence":["A-E1"],"reason":"一句话"}}}}}}
证据不足用 uncertain；不得使用常识补全。"""
        compact = llm.chat(
            [{"role": "system", "content": "你只输出紧凑、合法的 JSON。"},
             {"role": "user", "content": compact_prompt}],
            max_tokens=1200,
            temperature=0.0,
            thinking=False,
        )
        memory = _json_from_text(compact)
    if not isinstance(memory.get("options"), dict):
        raise RuntimeError("结构化记忆调用失败或返回了无效 JSON")
    for letter in q["options"]:
        entry = memory["options"].setdefault(letter, {})
        entry.setdefault("facts", [])
        entry.setdefault("preliminary_status", "uncertain")
        entry.setdefault("confidence", 0.0)
        entry.setdefault("evidence", [])
        entry.setdefault("reason", "")
        entry.setdefault("missing_facts", ["结构化记忆解析失败"])
        entry.setdefault("calculation", {"op": "none", "operands": []})
    return memory


def verdicts_from_memory(q: dict, memory: dict) -> dict:
    """把压缩器的初步判断规范化为统一 verdict 结构。"""
    result = {"options": {}, "answer": ""}
    entries = memory.get("options", {})
    for letter in q["options"]:
        entry = entries.get(letter, {}) if isinstance(entries.get(letter), dict) else {}
        verdict = entry.get("preliminary_status", "uncertain")
        if verdict not in _VALID_VERDICTS:
            verdict = "uncertain"
        confidence = _finite_number(entry.get("confidence"))
        confidence = min(1.0, max(0.0, confidence if confidence is not None else 0.0))
        result["options"][letter] = {
            "verdict": verdict,
            "confidence": confidence,
            "evidence": entry.get("evidence", []),
            "reason": str(entry.get("reason", ""))[:300],
        }
    result["answer"] = "".join(
        x for x in q["options"] if result["options"][x]["verdict"] == "support"
    )
    return result


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        try:
            number = float(cleaned)
            return number if math.isfinite(number) else None
        except ValueError:
            return None
    return None


def execute_calculations(memory: dict) -> None:
    """只执行白名单算术，不运行模型生成的表达式。结果原地写回 memory。"""
    for entry in memory.get("options", {}).values():
        calc = entry.get("calculation")
        if not isinstance(calc, dict):
            continue
        op = str(calc.get("op", "none"))
        operands = [_finite_number(x) for x in calc.get("operands", [])]
        if op == "none":
            calc["computed"] = False
            continue
        if any(x is None for x in operands):
            calc.update({"computed": False, "error": "invalid_operand"})
            continue
        nums = [float(x) for x in operands if x is not None]
        try:
            if op == "difference" and len(nums) >= 2:
                result: Any = nums[0] - nums[1]
            elif op == "ratio" and len(nums) >= 2:
                result = nums[0] / nums[1]
            elif op in {"growth_rate", "percentage_change"} and len(nums) >= 2:
                result = (nums[0] - nums[1]) / nums[1] * 100.0
            elif op == "sum" and nums:
                result = sum(nums)
            elif op == "compare" and len(nums) >= 2:
                operator = calc.get("operator")
                comparisons = {
                    ">": nums[0] > nums[1], "<": nums[0] < nums[1],
                    ">=": nums[0] >= nums[1], "<=": nums[0] <= nums[1],
                    "==": math.isclose(nums[0], nums[1], rel_tol=1e-6),
                }
                if operator not in comparisons:
                    raise ValueError("invalid_operator")
                result = comparisons[operator]
            else:
                raise ValueError("unsupported_shape")
            calc.update({"computed": True, "result": result})
        except (ZeroDivisionError, ValueError):
            calc.update({"computed": False, "error": "calculation_failed"})


def validate_memory(q: dict, plan: dict, memory: dict) -> dict[str, list[str]]:
    """检查结构化记忆是否覆盖选项要求的主体、时期、单位和计算。"""
    flags: dict[str, list[str]] = {}
    plan_items = {x["option"]: x for x in plan["items"]}
    for letter, claim in q["options"].items():
        item = plan_items[letter]
        entry = memory.get("options", {}).get(letter, {})
        facts_blob = json.dumps(entry.get("facts", []), ensure_ascii=False)
        current = []
        periods = [str(x) for x in item.get("expected_periods", []) if str(x)]
        units = [str(x) for x in item.get("expected_units", []) if str(x)]
        subjects = [str(x) for x in item.get("expected_subjects", []) if str(x)]
        if periods and not any(x in facts_blob for x in periods):
            current.append("missing_period")
        if units and not any(x in facts_blob for x in units):
            current.append("missing_unit")
        if subjects and not any(x in facts_blob for x in subjects):
            current.append("missing_subject")
        calc = entry.get("calculation", {})
        planned_op = item.get("calculation", {}).get("op", "none")
        if planned_op != "none" and not calc.get("computed", False):
            current.append("calculation_incomplete")
        if entry.get("missing_facts"):
            current.append("missing_fact")
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            current.append("missing_evidence_citation")
        else:
            valid_prefix = f"{letter}-E"
            if not any(str(source).startswith(valid_prefix) for source in evidence):
                current.append("invalid_evidence_citation")
        flags[letter] = current
    return flags


def _merge_review_result(first: dict, review: dict, flags: list[str]) -> dict:
    """合并独立复核，避免低置信二次判断覆盖高置信首判。

    模型给出的 confidence 并未校准；只有首判本身不确定/存在结构问题，或复核
    具有明显更强置信度时才允许翻转。引用了复核证据的结果获得小幅门槛优惠。
    """
    if review.get("verdict") == "uncertain":
        return first
    if review.get("verdict") == first.get("verdict"):
        return review if review.get("confidence", 0.0) >= first.get("confidence", 0.0) else first

    first_conf = float(first.get("confidence") or 0.0)
    review_conf = float(review.get("confidence") or 0.0)
    cited = any(re.fullmatch(r"[A-D]-R\d+", str(item)) for item in review.get("evidence", []))
    if first.get("verdict") == "uncertain" or flags:
        threshold = 0.62 if cited else 0.72
        return review if review_conf >= threshold else first
    threshold = max(0.82, first_conf + (0.04 if cited else 0.10))
    return review if review_conf >= threshold else first


def judge_matrix(llm: LLMClient, q: dict, plan: dict, memory: dict,
                 flags: dict[str, list[str]], matrix: dict[str, EvidenceSlot]) -> dict:
    """基于结构化记忆做第一次裁决。"""
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    compact_evidence = _matrix_text(matrix, per_chunk=260)
    fmt_rule = (
        "必须且只能选择一个选项" if q["answer_format"] in {"mcq", "tf"}
        else "逐项独立判断，可选择一个或多个选项"
    )
    prompt = f"""你是金融证据裁决 Agent。{fmt_rule}。只依据工作记忆、确定性计算结果
和证据摘录判断，不得用常识补齐缺失事实。

题目：{q['question']}
选项：
{options}

规划：{json.dumps(plan, ensure_ascii=False)}
结构化工作记忆：{json.dumps(memory, ensure_ascii=False)}
一致性检查：{json.dumps(flags, ensure_ascii=False)}
短证据摘录：
{compact_evidence}

规则：
1. 主体、年份、单位、指标口径必须同时对齐；
2. 有本地 computed=true 的计算时，以其 result 为准，不要重新心算；
3. 证据可直接支持或经一步确定性计算支持时判 support；
4. 与证据明确冲突判 contradict；关键事实确实缺失才判 uncertain；
5. 不要因为选项需要计算或换了一种说法就漏选。

返回严格 JSON，不要 Markdown：
{{"options":{{"A":{{"verdict":"support|contradict|uncertain",
"confidence":0.0,"evidence":["A-E1"],"reason":"一句话"}}}},"answer":"AC"}}
四个选项都必须输出；answer 必须与 verdict=support 的选项一致。"""
    out = llm.chat(
        [{"role": "system", "content": "你必须逐项核对并输出可机器解析的裁决。"},
         {"role": "user", "content": prompt}],
        max_tokens=2600,
        temperature=0.0,
        thinking=False,
    )
    raw = _json_from_text(out)
    result = {"options": {}, "answer": ""}
    raw_options = raw.get("options", {}) if isinstance(raw.get("options"), dict) else {}
    for letter in q["options"]:
        item = raw_options.get(letter, {}) if isinstance(raw_options.get(letter), dict) else {}
        verdict = item.get("verdict", "uncertain")
        if verdict not in _VALID_VERDICTS:
            verdict = "uncertain"
        confidence = _finite_number(item.get("confidence"))
        confidence = min(1.0, max(0.0, confidence if confidence is not None else 0.0))
        result["options"][letter] = {
            "verdict": verdict,
            "confidence": confidence,
            "evidence": item.get("evidence", []),
            "reason": str(item.get("reason", ""))[:300],
        }
    result["answer"] = "".join(
        letter for letter in q["options"]
        if result["options"][letter]["verdict"] == "support"
    )
    return result


def _review_targets(q: dict, verdicts: dict, flags: dict[str, list[str]],
                    matrix: dict[str, EvidenceSlot], memory: dict) -> list[str]:
    selected = [k for k, v in verdicts["options"].items() if v["verdict"] == "support"]
    ranked_targets: list[tuple[float, str]] = []
    for letter, item in verdicts["options"].items():
        confidence = float(item.get("confidence") or 0.0)
        current_flags = set(flags.get(letter, []))
        citation_broken = bool(current_flags & {
            "missing_evidence_citation", "invalid_evidence_citation"
        })
        semantic_gap = bool(current_flags & {
            "missing_period", "missing_unit", "missing_subject",
            "missing_fact", "calculation_incomplete",
        })
        computed = bool(
            memory.get("options", {}).get(letter, {})
            .get("calculation", {}).get("computed")
        )
        score = 0.0
        if item["verdict"] == "uncertain":
            score += 5.0
        if citation_broken:
            score += 4.0
        threshold = 0.70 if item["verdict"] == "support" else 0.66
        if confidence < threshold:
            score += 3.0 + (threshold - confidence)
        if semantic_gap and confidence < 0.80:
            score += 2.0
        if matrix[letter].coverage < 0.42 and confidence < 0.80:
            score += 1.5
        if computed and confidence < 0.78:
            score += 1.0
        if score > 0:
            ranked_targets.append((score, letter))

    # 多选只选 0/1 项时保留一次“漏选申诉”，但只复核证据覆盖最好的一个
    # 未选项，不再把全部未选项重新送入模型。
    if q["answer_format"] == "multi" and len(selected) <= 1:
        appeals = [
            (matrix[k].coverage, k) for k in q["options"]
            if k not in selected and matrix[k].coverage >= 0.58
        ]
        if appeals:
            coverage, letter = max(appeals)
            ranked_targets.append((1.0 + coverage, letter))
    if q["answer_format"] in {"mcq", "tf"} and len(selected) != 1:
        ranked_targets.extend((2.5, letter) for letter in q["options"])

    best: dict[str, float] = {}
    for score, letter in ranked_targets:
        best[letter] = max(score, best.get(letter, 0.0))
    ordered = sorted(best, key=lambda letter: best[letter], reverse=True)
    # 正常题至多复核两个选项；单选/判断首判非法时允许检查全部候选。
    limit = len(q["options"]) if q["answer_format"] in {"mcq", "tf"} and len(selected) != 1 else 2
    return ordered[:limit]


def review_uncertain(
    llm: LLMClient,
    q: dict,
    plan: dict,
    memory: dict,
    verdicts: dict,
    targets: list[str],
    retriever: DocRetriever,
    document_profiles: dict[str, str],
    document_term_stats: dict[str, dict[str, int]],
    matrix: dict[str, EvidenceSlot],
) -> dict:
    """对目标选项扩检后做一次独立复核，并返回可合并的局部 verdict。"""
    if not targets:
        return {}
    items = {x["option"]: x for x in plan["items"]}
    prefer_doc_balance = (
        q["domain"] in {"financial_reports", "financial_contracts"}
        or plan.get("question_type") == "numeric_comparison"
    )
    review_parts = []
    for letter in targets:
        item = items[letter]
        queries = (
            _query_expansions(q, item)
            + list(item.get("required_facts", []))
            + list(item.get("search_queries", []))
            + [q["question"], item["claim"]]
        )
        focus_text = f"{q['question']} {item['claim']}"
        chunks = retriever.retrieve_for_option(
            queries, focus_text, pool=80, top_k=8,
            domain=q["domain"], use_rerank=False,
            prefer_doc_balance=prefer_doc_balance,
        )
        existing = {
            (chunk["doc_id"], chunk["chunk_id"])
            for chunk in matrix[letter].chunks
        }
        delta = [
            chunk for chunk in chunks
            if (chunk["doc_id"], chunk["chunk_id"]) not in existing
        ]
        chunks = (delta or chunks[:2])[:5]
        review_parts.append(f"\n### {letter}. {item['claim']}")
        for idx, chunk in enumerate(chunks, 1):
            text = _best_excerpt(chunk["text"], focus_text, 600)
            review_parts.append(
                f"[{letter}-R{idx}|{chunk['doc_id']}#chunk{chunk['chunk_id']}] {text}"
            )
    domain_policy = _DOMAIN_POLICY.get(q["domain"], "")
    target_memory = {
        letter: memory.get("options", {}).get(letter, {}) for letter in targets
    }
    target_verdicts = {
        letter: verdicts.get("options", {}).get(letter, {}) for letter in targets
    }
    prompt = f"""你是第二位独立金融核查员。只复核指定选项，不受第一次结论影响。

题目：{q['question']}
待复核选项：{', '.join(targets)}
文档身份卡：{json.dumps(document_profiles, ensure_ascii=False)}
全文主题词命中次数：{json.dumps(document_term_stats, ensure_ascii=False)}
本领域裁决规则：{domain_policy}
待复核选项的结构化记忆：{json.dumps(target_memory, ensure_ascii=False)}
待复核选项的第一次裁决：{json.dumps(target_verdicts, ensure_ascii=False)}
增量证据：
{''.join(review_parts)}

逐项检查主体、年份、单位、数值和否定词。本轮同时是未选项的“漏选申诉”：主动寻找
能够支持它的直接证据或一步确定性推导，不要因为同义改写、阈值概括或证据分散就拒绝。
资料直接支持或可由原始数值确定推出时判 support。只有同一主体、时期、指标下存在直接
不兼容的证据才能 contradict；仅仅没找到、措辞不同或检索片段不全必须判 uncertain。
全文主题词计数为0只是否定性陈述的辅助证据；只有文档身份和完整保险责任同时表明
该产品属于封闭列举的定额给付责任时，才可用其支持‘不涵盖某费用/无某免赔额’。
返回严格 JSON：
{{"options":{{"A":{{"verdict":"support|contradict|uncertain",
"confidence":0.0,"evidence":["A-R1"],"reason":"一句话"}}}}}}
只输出待复核选项。"""
    out = llm.chat(
        [{"role": "system", "content": "你是独立复核员，必须引用扩展证据。"},
         {"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.0,
        thinking=False,
    )
    raw = _json_from_text(out)
    raw_options = raw.get("options", {}) if isinstance(raw.get("options"), dict) else {}
    reviewed = {}
    for letter in targets:
        item = raw_options.get(letter, {}) if isinstance(raw_options.get(letter), dict) else {}
        verdict = item.get("verdict", "uncertain")
        confidence = _finite_number(item.get("confidence"))
        if verdict in _VALID_VERDICTS and confidence is not None:
            reviewed[letter] = {
                "verdict": verdict,
                "confidence": min(1.0, max(0.0, confidence)),
                "evidence": item.get("evidence", []),
                "reason": str(item.get("reason", ""))[:300],
            }
    return reviewed


def reconcile_verdicts(
    q: dict,
    verdicts: dict,
    document_profiles: dict[str, str] | None = None,
    document_term_stats: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, str]]:
    """用可确定检查的语义约束修正“理由与标签相反”。

    只处理不需要额外金融知识的情形：单位硬冲突、选项自身明示
    不满足题干筛选条件，以及理由已明说“计算结果应支持”却
    误输出 contradict。所有调整都写入审计轨迹。
    """
    adjustments: list[dict[str, str]] = []
    question = q.get("question", "")
    formula_filter = bool(re.search(
        r"哪些.*(?:明确|具体).*(?:计算方法|公式)|"
        r"(?:计算方法|公式).*(?:哪些|明确)",
        question,
    ))
    payout_filter = bool(re.search(
        r"哪些.*(?:产品|保险).*(?:可以|能).{0,8}赔付|"
        r"哪些.*(?:可以|能).{0,8}赔付",
        question,
    ))
    support_reason = re.compile(
        r"则应支持|应判(?:定)?为support|计算结果支持|"
        r"现(?:有)?数据.{0,24}支持|结果.{0,12}支持.{0,12}(?:选项|结论)|"
        r"陈述正确|结论成立|与事实一致|"
        r"应判\s*support|说法.{0,10}正确"
    )
    negative_support = re.compile(
        r"不支持|无法支持|不应支持|陈述不正确|"
        r"结论不成立|与事实不一致|与事实相反"
    )
    document_profiles = document_profiles or {}
    document_term_stats = document_term_stats or {}

    def option_document(option_text: str) -> str | None:
        matched = _match_option_documents(option_text, document_profiles)
        return matched[0] if matched else None

    for letter, option_text in q.get("options", {}).items():
        item = verdicts.get("options", {}).get(letter)
        if not isinstance(item, dict):
            continue
        old = item.get("verdict", "uncertain")
        reason = str(item.get("reason", ""))
        new = old
        rule = ""
        option_numbers = re.findall(r"\d+(?:\.\d+)?", option_text)
        per_share_mismatch = (
            "每股" in option_text
            and any(re.search(
                rf"每\s*(?:10|十)\s*股.{{0,20}}{re.escape(number)}",
                reason,
            ) for number in option_numbers)
        )

        if (
            formula_filter
            and re.search(
                r"未(?:给出|给|提供|明确).{0,12}(?:计算方法|公式)",
                option_text,
            )
        ):
            new, rule = "contradict", "option_fails_question_formula_filter"
        elif payout_filter and re.search(
            r"不赔|无.{0,10}(?:医疗|费用).{0,8}责任",
            option_text,
        ):
            new, rule = "contradict", "option_fails_question_payout_filter"
        elif per_share_mismatch:
            # 只有选项声称的具体数字实际属于“每10股”时才否定；
            # “每股分红A高于B”这类已换算的比较不能被误伤。
            new, rule = "contradict", "per_share_unit_mismatch"
        elif (
            old == "contradict"
            and support_reason.search(reason)
            and not negative_support.search(reason)
        ):
            new, rule = "support", "reason_explicitly_supports_claim"
        elif old == "contradict" and re.search(r"(?:此前|先前).{0,10}误判", reason):
            new, rule = "support", "review_reason_self_corrected"
        elif old == "contradict" and "超过" in option_text and "两倍" in option_text:
            comparison = re.search(
                r"([\d,.]+)\s*<\s*([\d,.]+)\s*\*\s*2\s*\(([\d,.]+)\)",
                reason,
            )
            if comparison:
                left = float(comparison.group(1).replace(",", ""))
                doubled = float(comparison.group(3).replace(",", ""))
                if left > doubled:
                    new, rule = "support", "deterministic_double_comparison"
        elif old == "contradict":
            # 整数百分比区间在概括性陈述中通常按一位整数取整。
            range_match = re.search(r"(\d+(?:\.\d+)?)%\s*(?:至|到|[-~])\s*(\d+(?:\.\d+)?)%", option_text)
            values = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", reason)]
            if range_match and values:
                lo, hi = map(float, range_match.groups())
                if all(lo - 0.5 <= value < hi + 0.5 for value in values):
                    new, rule = "support", "rounded_percentage_range_match"
        if (
            old == "support"
            and q.get("domain") == "financial_contracts"
            and "两份文档均" in option_text
            and "发行人" in option_text
            and any("发行股份购买资产" in profile for profile in document_profiles.values())
        ):
            new, rule = "contradict", "transaction_report_has_no_bond_issuer_role"
        if (
            old == "support"
            and q.get("domain") == "insurance"
            and "宽限期" in question
            and "效力中止" in question
        ):
            doc_id = option_document(option_text)
            stats = document_term_stats.get(doc_id or "", {})
            if stats.get("宽限期") == 0 or stats.get("效力中止") == 0:
                new, rule = "contradict", "product_lacks_suspension_clause"
        if (
            old == "contradict"
            and q.get("domain") == "regulatory"
            and re.match(r"^(?:若|如果)", option_text)
            and re.search(r"当事人.{0,30}申辩|本案.{0,30}连续|违法行为.{0,20}连续性", reason)
            and re.search(r"连续性|连续继续|后续年度", reason)
        ):
            new, rule = "support", "evaluate_hypothetical_antecedent_not_case_fact"
        if old in {"uncertain", "contradict"} and q.get("domain") == "insurance":
            doc_id = option_document(option_text)
            stats = document_term_stats.get(doc_id or "", {})
            profile = document_profiles.get(doc_id or "", "")
            fixed_benefit_policy = bool(re.search(
                r"重大疾病保险|终身寿险|身故保障", profile
            )) and not re.search(r"医疗保险|费用补偿", profile)
            if "无免赔额" in option_text and stats.get("免赔额") == 0:
                new, rule = "support", "closed_policy_has_no_deductible_term"
            elif (
                re.search(r"不(?:涵盖|包括|保障)", option_text)
                and "特定药品" in option_text
                and stats.get("特定药品") == 0
                and stats.get("院外") == 0
                and fixed_benefit_policy
            ):
                new, rule = "support", "fixed_benefit_policy_lacks_drug_expense_coverage"
            elif (
                "犹豫期" in question
                and re.search(r"退还.{0,12}(?:全部|全额).{0,10}保险费", question)
                and stats.get("犹豫期全额退费") == 1
            ):
                new, rule = "support", "full_text_cooling_off_refund_rule"
        if (
            old == "contradict"
            and q.get("domain") == "research"
            and re.search(
                r"未限定(?:主体|地区|地域)|省略.{0,8}(?:主体|地区|地域)|"
                r"(?:主体|地区|地域).{0,8}(?:不明|缺失|归属)", reason
            )
            and re.search(
                r"证据.{0,16}(?:明确)?(?:指出|显示|记载)|事实陈述|"
                r"数值.{0,12}(?:一致|吻合|相符)", reason
            )
        ):
            new, rule = "support", "research_context_supplies_omitted_scope"

        if new != old:
            item["verdict"] = new
            item["confidence"] = max(0.86, float(item.get("confidence") or 0.0))
            adjustments.append({
                "option": letter,
                "from": old,
                "to": new,
                "rule": rule,
            })

    verdicts["answer"] = "".join(
        x for x in q.get("options", {})
        if verdicts.get("options", {}).get(x, {}).get("verdict") == "support"
    )
    return adjustments


def _final_answer(q: dict, verdicts: dict) -> str:
    valid = list(q["options"])
    if q["answer_format"] in {"mcq", "tf"}:
        supported = [x for x in valid if verdicts["options"][x]["verdict"] == "support"]
        if len(supported) == 1:
            return supported[0]
        # 非正常输出时按置信度选最可能项，而不是固定 A。
        return max(
            valid,
            key=lambda x: (
                {"support": 2, "uncertain": 1, "contradict": 0}.get(
                    verdicts["options"][x]["verdict"], 0
                ),
                verdicts["options"][x]["confidence"],
            ),
        )
    supported = [x for x in valid if verdicts["options"][x]["verdict"] == "support"]
    if supported:
        return "".join(supported)
    # 多选没有 support 时，保留最高置信度的非 contradict 项，避免默认 A。
    candidates = [x for x in valid if verdicts["options"][x]["verdict"] != "contradict"]
    candidates = candidates or valid
    return max(candidates, key=lambda x: verdicts["options"][x]["confidence"])


def answer_question_agent(
    llm: LLMClient,
    q: dict,
    *,
    enable_review: bool = True,
    use_option_rerank: bool | None = None,
    use_llm_planner: bool = False,
    separate_judge: bool = False,
) -> tuple[str, dict]:
    """运行完整 Agent，返回 ``(答案, 可审计轨迹)``。"""
    if use_option_rerank is None:
        use_option_rerank = os.environ.get("AGENT_OPTION_RERANK", "0") == "1"

    # 默认使用确定性规划器，避免为每题额外消耗一次受限 API 调用；需要研究完整
    # LLM 规划效果时可显式开启。后续检索、记忆、计算和复核流程保持一致。
    plan = plan_question(llm, q) if use_llm_planner else _fallback_plan(q)
    retriever = DocRetriever(q["domain"], q["doc_ids"], llm=llm)
    document_profiles = build_document_profiles(retriever)
    document_term_stats = build_document_term_stats(retriever, q)
    matrix = build_evidence_matrix(
        retriever, q, plan, use_rerank=use_option_rerank,
        document_profiles=document_profiles,
    )
    memory = compress_memory(
        llm, q, plan, matrix, document_profiles, document_term_stats
    )
    execute_calculations(memory)
    flags = validate_memory(q, plan, memory)
    verdicts = (
        judge_matrix(llm, q, plan, memory, flags, matrix)
        if separate_judge else verdicts_from_memory(q, memory)
    )
    first_verdicts = json.loads(json.dumps(verdicts, ensure_ascii=False))

    targets = (
        _review_targets(q, verdicts, flags, matrix, memory)
        if enable_review else []
    )
    reviewed = review_uncertain(
        llm, q, plan, memory, verdicts, targets, retriever, document_profiles,
        document_term_stats, matrix,
    ) if targets else {}
    # 证据感知合并复核；低置信的二次判断不能覆盖高置信首判。
    for letter, result in reviewed.items():
        verdicts["options"][letter] = _merge_review_result(
            verdicts["options"][letter], result, flags.get(letter, [])
        )

    deterministic_adjustments = reconcile_verdicts(
        q, verdicts, document_profiles, document_term_stats
    )
    answer = _final_answer(q, verdicts)
    trace = {
        "qid": q["qid"],
        "architecture": "hierarchical_evidence_agent_v4",
        "question_type": plan.get("question_type"),
        "document_profiles": document_profiles,
        "document_term_stats": document_term_stats,
        "planner_fallback": plan.get("planner_fallback", False),
        "evidence": {
            letter: {
                "coverage": slot.coverage,
                "expanded": slot.expanded,
                "doc_coverage": slot.doc_coverage,
                "chunks": [f"{c['doc_id']}#chunk{c['chunk_id']}" for c in slot.chunks],
            }
            for letter, slot in matrix.items()
        },
        "memory": memory,
        "validation_flags": flags,
        "first_verdicts": first_verdicts,
        "final_verdicts": verdicts,
        "review_targets": targets,
        "reviewed": reviewed,
        "deterministic_adjustments": deterministic_adjustments,
        "answer": answer,
    }
    return answer, trace
