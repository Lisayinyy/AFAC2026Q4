"""无需模型的轻量语义分段。

金融文档常把同一指标的定义、数值和比较结论拆成相邻句。按固定字符截断会
破坏这种关系；这里用词汇重叠、数字和金融桥接词寻找主题边界，不产生任何
推理阶段模型调用，也不依赖 embedding。
"""
from __future__ import annotations

import math
import re
from typing import Callable


_SENTENCE_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
_BRIDGE_TERMS = {
    "同比", "环比", "增长", "下降", "上升", "增加", "减少", "变化",
    "占比", "比例", "率", "金额", "规模", "年度", "季度", "平均",
    "净利润", "营业收入", "现金流", "资产负债", "研发投入", "分红",
    "保险责任", "免赔额", "等待期", "宽限期", "现金价值", "发行人",
}


def _terms(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z0-9.%]+|[\u4e00-\u9fff]{2,}", text)
    out: set[str] = set(raw)
    for item in raw:
        if len(item) > 4 and all("\u4e00" <= c <= "\u9fff" for c in item):
            out.update(item[i:i + 2] for i in range(len(item) - 1))
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _bridge_score(left: str, right: str) -> float:
    shared_terms = {term for term in _BRIDGE_TERMS if term in left and term in right}
    shared_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", left)) & set(
        re.findall(r"\d+(?:\.\d+)?%?", right)
    )
    if shared_terms:
        return min(0.45, 0.22 * len(shared_terms))
    if shared_numbers:
        return 0.12
    return 0.0


def split_semantic(
    paragraph: str,
    *,
    max_sentences: int = 8,
    threshold: float = 0.08,
    similarity: Callable[[str, str], float] | None = None,
) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.findall(paragraph) if s.strip()]
    if len(sentences) <= 1:
        return sentences or [paragraph.strip()]
    vectors = [_terms(sentence) for sentence in sentences]
    groups: list[str] = []
    current = [sentences[0]]
    for index in range(1, len(sentences)):
        score = (
            similarity(sentences[index - 1], sentences[index])
            if similarity
            else _jaccard(vectors[index - 1], vectors[index])
            + _bridge_score(sentences[index - 1], sentences[index])
        )
        if len(current) >= max_sentences or score < threshold:
            groups.append("".join(current))
            current = []
        current.append(sentences[index])
    if current:
        groups.append("".join(current))
    return groups
