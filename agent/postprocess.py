# 答案后处理：字母抽取、排序去重、合法性校验（评测严格匹配，无部分分）
import re

VALID = set("ABCD")


def normalize_answer(raw: str, answer_format: str) -> str:
    """从模型输出中抽取合法答案字符串。

    - mcq: 取首个有效字母(A-D)
    - tf: 只认 A/B
    - multi: 去重 + 按字母序拼接（评测要求如 "ABC" 无分隔符）
    """
    if not raw:
        return ""
    valid = set("AB") if answer_format == "tf" else VALID
    # 优先找 "答案：X" / "最终答案" 附近的字母，否则全局抽取
    m = re.search(r"(?:最终答案|答案)[:：\s]*([A-D][A-D、,，\s]*)", raw)
    segment = m.group(1) if m else raw
    letters = [ch for ch in segment.upper() if ch in valid]
    if not letters:
        letters = [ch for ch in raw.upper() if ch in valid]
    if not letters:
        return ""
    if answer_format in ("mcq", "tf"):
        return letters[0]
    # multi
    return "".join(sorted(set(letters)))
