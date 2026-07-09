# 单题答题管线 V1：BM25 召回 -> qwen-flash 精排 -> 分题型答题（多选逐选项验证）
import json
import re

from . import config
from .qwen_client import chat
from .retrieve import get_index
from .postprocess import normalize_answer

FORMAT_INSTR = {
    "mcq": "这是单选题，只有一个正确选项。回答的第一行必须是：答案：X（X 为 A/B/C/D 中的一个字母），然后另起一行给出不超过150字的依据。",
    "multi": "这是多选题，可能有多个正确选项。回答的第一行必须是：答案：XY（所有正确选项字母，按字母顺序，无分隔符），然后另起一行给出每个选项不超过50字的判断依据。注意漏选/多选均不得分，务必严格依据原文证据判断每个选项。",
    "tf": "这是判断题。回答的第一行必须是：答案：A 或 答案：B（含义以题目选项为准），然后另起一行给出不超过100字的依据。",
}

DOMAIN_HINTS = {
    "insurance": "注意保险条款中的触发条件、计算公式、等待期、免责条款和例外情形。涉及金额计算时先列出公式和各变量取值再逐步计算。",
    "regulatory": "严格依据法规条文作答，注意施行日期、适用范围、时限要求和条文优先级，不得用常识替代条文。引用条文编号。",
    "financial_contracts": "注意债券条款中的期限、利率、担保、评级、赎回/回售条件及权利义务主体。",
    "financial_reports": "注意会计科目口径、报告期与同比基期，比较类题目需分别列出两期具体数值再判断增减，禁止凭印象。",
    "research": "严格依据研报原文的结论与数据，区分作者观点与引用数据，注意预测值与实际值。",
}


def _evidence_text(chunks):
    return "\n\n".join(
        f"【证据{i + 1}｜{c['doc_id']}】\n{c['text']}" for i, c in enumerate(chunks)
    )


# ---------- L3: LLM 精排 ----------

def rerank(q, candidates, qid=None):
    """qwen-flash 对候选块批量打分 0-2，保留高分块；失败时回退 BM25 顺序。"""
    if not config.RERANK_ENABLED or len(candidates) <= config.RERANK_KEEP:
        return candidates[: config.RERANK_KEEP]
    listing = "\n\n".join(
        f"[{i}] ({c['doc_id']}) {c['text'][:350]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"问题：{q['question']}\n选项：{json.dumps(q['options'], ensure_ascii=False)}\n\n"
        f"下面是候选文档片段，请判断每个片段对回答该问题的价值：2=直接包含答案证据，1=相关背景，0=无关。\n"
        f"{listing}\n\n"
        f"只输出 JSON，格式：{{\"scores\": {{\"0\": 2, \"1\": 0, ...}}}}，包含所有编号。"
    )
    try:
        raw = chat([{"role": "user", "content": prompt}],
                   model=config.LITE_MODEL, qid=qid, max_tokens=800)
        m = re.search(r"\{.*\}", raw, re.S)
        scores = json.loads(m.group(0))["scores"] if m else {}
        ranked = sorted(range(len(candidates)),
                        key=lambda i: (-int(scores.get(str(i), 0)), i))
        keep = [candidates[i] for i in ranked if int(scores.get(str(i), 0)) > 0]
        if len(keep) < 3:  # 精排过狠时补回 BM25 头部
            keep = (keep + [c for c in candidates if c not in keep])[: config.RERANK_KEEP]
        return keep[: config.RERANK_KEEP]
    except Exception:
        return candidates[: config.RERANK_KEEP]


# ---------- 文档路由（B 榜 / doc_ids 缺失时） ----------

def route_docs(q, index):
    if q.get("doc_ids"):
        return q["doc_ids"]
    hits = index.search(q["question"] + " " + " ".join(q["options"].values()),
                        top_k=config.BM25_CANDIDATES)
    doc_scores = {}
    for rank, c in enumerate(hits):
        doc_scores[c["doc_id"]] = doc_scores.get(c["doc_id"], 0) + (config.BM25_CANDIDATES - rank)
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:4]]


# ---------- 多选：逐选项验证 ----------

def answer_multi_by_option(q, evidence, qid):
    """每个选项独立判断，降低整题连坐概率。返回如 'ABC'；全否时回退整体作答。"""
    ev = _evidence_text(evidence)
    system = "你是金融文档问答专家。仅依据给出的文档证据判断。" + DOMAIN_HINTS.get(q["domain"], "")
    picked = []
    for letter, opt in q["options"].items():
        user = (
            f"参考文档证据：\n{ev}\n\n"
            f"问题背景：{q['question']}\n\n"
            f"待判断陈述（选项{letter}）：{opt}\n\n"
            f"该陈述根据证据是否准确/正确？回答的第一行必须是：判断：正确 或 判断：错误，"
            f"然后另起一行给出不超过80字的证据依据。证据不足以支持该陈述时判断为错误。"
        )
        raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                   model=config.ANSWER_MODEL, qid=qid, max_tokens=400)
        head = raw.strip()[:30]
        if "判断：正确" in head or head.startswith("正确"):
            picked.append(letter)
    if picked:
        return "".join(sorted(picked))
    return None  # 触发整体作答回退


# ---------- 主入口 ----------

def build_prompt(q, evidence_chunks):
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    system = (
        "你是金融文档问答专家。仅依据给出的文档证据回答，证据不足时选择最可能的答案。"
        + DOMAIN_HINTS.get(q["domain"], "")
    )
    user = (
        f"参考文档证据：\n{_evidence_text(evidence_chunks)}\n\n"
        f"问题：{q['question']}\n\n选项：\n{options}\n\n"
        f"{FORMAT_INSTR[q['answer_format']]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_question(q):
    qid = q["qid"]
    index = get_index(q["domain"])
    doc_ids = route_docs(q, index)
    query = q["question"] + " " + " ".join(q["options"].values())
    candidates = index.search(query, top_k=config.BM25_CANDIDATES, doc_ids=doc_ids)
    evidence = rerank(q, candidates, qid=qid)

    ans, raw = "", ""
    if q["answer_format"] == "multi" and config.MULTI_VERIFY:
        ans = answer_multi_by_option(q, evidence, qid) or ""
    if not ans:
        raw = chat(build_prompt(q, evidence), model=config.ANSWER_MODEL, qid=qid,
                   max_tokens=1200)
        ans = normalize_answer(raw, q["answer_format"])
    if not ans:  # 最终兜底，不留空（空=必错，蒙也要填）
        ans = "A" if q["answer_format"] in ("mcq", "tf") else "AB"
    return {
        "qid": qid,
        "answer": ans,
        "raw": raw,
        "doc_ids_used": doc_ids,
        "evidence_ids": [c["chunk_id"] for c in evidence],
    }
