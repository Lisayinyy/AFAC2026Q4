# 检索：BM25 + 关键词加权混合（赛规禁止 embedding 模型，纯词法检索合规）
import re
import jieba
from rank_bm25 import BM25Okapi

from . import config

_STOP = set("的了在是和与及或对于根据以下关于按照有关其中进行相关如下所述本各该之为".split())


def tokenize(text: str):
    return [t for t in jieba.lcut(text) if t.strip() and t not in _STOP and len(t.strip()) > 0]


class DomainIndex:
    """一个领域一个索引。支持限定 doc_ids 检索（A榜）与全域检索（B榜）。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self._tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens) if chunks else None

    def search(self, query: str, top_k=None, doc_ids=None):
        top_k = top_k or config.TOP_K_CHUNKS
        if not self.bm25:
            return []
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        # 精确串加权：条款号（第X条）、百分比、金额、年份等命中直接加分
        patterns = re.findall(r"第[一二三四五六七八九十百零\d]+条|\d+(?:\.\d+)?%|\d{4}年|\d+(?:\.\d+)?[万亿]元?", query)
        results = []
        allowed = set(doc_ids) if doc_ids else None
        for i, c in enumerate(self.chunks):
            if allowed and c["doc_id"] not in allowed:
                continue
            bonus = sum(2.0 for p in patterns if p in c["text"])
            results.append((scores[i] + bonus, c))
        results.sort(key=lambda x: -x[0])
        return [c for s, c in results[:top_k]]


_INDEX_CACHE = {}


def get_index(domain: str) -> DomainIndex:
    if domain not in _INDEX_CACHE:
        from .parse import load_chunks
        _INDEX_CACHE[domain] = DomainIndex(load_chunks(domain))
    return _INDEX_CACHE[domain]
