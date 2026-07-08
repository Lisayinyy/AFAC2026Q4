# Qwen API 客户端：仅走百炼 OpenAI 兼容接口，带全局 token 统计
import os
import time
import json
import threading
from openai import OpenAI

from . import config


class TokenLedger:
    """全局 token 账本，评测要求提交 prompt/completion/total 统计。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.per_qid = {}  # qid -> {"prompt_tokens":..,"completion_tokens":..}

    def add(self, usage, qid=None):
        with self._lock:
            pt = usage.prompt_tokens or 0
            ct = usage.completion_tokens or 0
            self.prompt_tokens += pt
            self.completion_tokens += ct
            if qid:
                rec = self.per_qid.setdefault(qid, {"prompt_tokens": 0, "completion_tokens": 0})
                rec["prompt_tokens"] += pt
                rec["completion_tokens"] += ct

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def summary(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "per_qid": self.per_qid}, f,
                      ensure_ascii=False, indent=2)


LEDGER = TokenLedger()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get(config.API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"请先设置环境变量 {config.API_KEY_ENV}")
        _client = OpenAI(api_key=api_key, base_url=config.DASHSCOPE_BASE_URL)
    return _client


def chat(messages, model=None, qid=None, max_retries=3, **kwargs):
    """带重试与 token 记账的对话调用。返回 message content 字符串。"""
    model = model or config.ANSWER_MODEL
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.pop("temperature", 0.0),
                **kwargs,
            )
            if resp.usage:
                LEDGER.add(resp.usage, qid=qid)
            return resp.choices[0].message.content
        except Exception as e:  # 限流/网络错误退避重试
            last_err = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"Qwen API 调用失败: {last_err}")
