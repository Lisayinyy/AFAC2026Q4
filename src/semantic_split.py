"""轻量语意分词模块。

不强制下载模型：默认用句子之间的词汇相似度寻找主题切换点；部署时可以
通过 ``SemanticSegmenter`` 接入外部语意模型，只需返回相邻句子的相似度，
下游 chunk 格式无需改变。这样满足赛题允许外部分词模型的扩展点，同时保持
离线可复现和低 Token 成本。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable


_SENTENCE_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")


def _terms(text: str) -> set[str]:
    # 中文按连续字母/数字和双字切分，兼顾公司名、条款号、百分比。
    raw = re.findall(r"[A-Za-z0-9.%]+|[\u4e00-\u9fff]{2,}", text)
    out: set[str] = set(raw)
    for item in raw:
        if len(item) > 4 and all("\u4e00" <= c <= "\u9fff" for c in item):
            out.update(item[i : i + 2] for i in range(len(item) - 1))
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def split_semantic(
    paragraph: str,
    *,
    max_sentences: int = 8,
    threshold: float = 0.08,
    similarity: Callable[[str, str], float] | None = None,
) -> list[str]:
    """按语意相似度将段落拆成主题连贯的小组。

    ``similarity`` 可由外部模型注入；没有注入时采用词汇 cohesion。低于
    threshold 的相邻句子会形成边界，但单组不会超过 max_sentences。
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(paragraph) if s.strip()]
    if len(sentences) <= 1:
        return sentences or [paragraph.strip()]
    vectors = [_terms(s) for s in sentences]
    groups: list[str] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = similarity(sentences[i - 1], sentences[i]) if similarity else _jaccard(vectors[i - 1], vectors[i])
        if len(current) >= max_sentences or sim < threshold:
            groups.append("".join(current))
            current = []
        current.append(sentences[i])
    if current:
        groups.append("".join(current))
    return groups

