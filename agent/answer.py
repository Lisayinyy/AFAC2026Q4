# 单题答题管线：检索证据 -> 构造 prompt -> Qwen 答题 -> 答案规范化
import json

from . import config
from .qwen_client import chat
from .retrieve import get_index
from .postprocess import normalize_answer

FORMAT_INSTR = {
    "mcq": "这是单选题，只有一个正确选项。最后一行必须输出：答案：X（X 为 A/B/C/D 中的一个字母）",
    "multi": "这是多选题，可能有多个正确选项。逐个选项判断对错后，最后一行必须输出：答案：XY（所有正确选项字母，按字母顺序，无分隔符）。注意漏选/多选均不得分，务必严格依据原文证据判断每个选项。",
    "tf": "这是判断题。最后一行必须输出：答案：A 或 答案：B（含义以选项为准）",
}

DOMAIN_HINTS = {
    "insurance": "注意保险条款中的触发条件、计算公式、等待期、免责条款和例外情形。涉及金额计算时先列出公式和各变量取值再计算。",
    "regulatory": "严格依据法规条文作答，注意施行日期、适用范围、时限要求和条文优先级，不得用常识替代条文。",
    "financial_contracts": "注意债券条款中的期限、利率、担保、评级、赎回/回售条件及权利义务主体。",
    "financial_reports": "注意会计科目口径、报告期与同比基期，比较类题目需分别核对两期数值再判断增减。",
    "research": "严格依据研报原文的结论与数据，区分作者观点与引用数据，注意预测值与实际值。",
}


def build_prompt(q: dict, evidence_chunks):
    options = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    evidence = "\n\n".join(
        f"【证据{i + 1}｜{c['doc_id']}】\n{c['text']}" for i, c in enumerate(evidence_chunks)
    )
    system = (
        "你是金融文档问答专家。仅依据给出的文档证据回答，证据不足时选择最可能的答案。"
        + DOMAIN_HINTS.get(q["domain"], "")
    )
    user = (
        f"参考文档证据：\n{evidence}\n\n"
        f"问题：{q['question']}\n\n选项：\n{options}\n\n"
        f"{FORMAT_INSTR[q['answer_format']]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def route_docs(q: dict, index) -> list:
    """B 榜：题目不给 doc_ids，先用轻量模型 + BM25 路由候选文档。"""
    if q.get("doc_ids"):
        return q["doc_ids"]
    # 先 BM25 全域检索取候选块，统计其所属 doc 的得分
    hits = index.search(q["question"] + " " + " ".join(q["options"].values()),
                        top_k=config.BM25_CANDIDATES)
    doc_scores = {}
    for rank, c in enumerate(hits):
        doc_scores[c["doc_id"]] = doc_scores.get(c["doc_id"], 0) + (config.BM25_CANDIDATES - rank)
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:4]]


def answer_question(q: dict) -> dict:
    index = get_index(q["domain"])
    doc_ids = route_docs(q, index)
    query = q["question"] + " " + " ".join(q["options"].values())
    evidence = index.search(query, top_k=config.TOP_K_CHUNKS, doc_ids=doc_ids)
    messages = build_prompt(q, evidence)
    raw = chat(messages, model=config.ANSWER_MODEL, qid=q["qid"])
    ans = normalize_answer(raw, q["answer_format"])
    return {
        "qid": q["qid"],
        "answer": ans,
        "raw": raw,
        "doc_ids_used": doc_ids,
        "evidence_ids": [c["chunk_id"] for c in evidence],
    }
