# 文档预处理：PDF/txt -> 结构化纯文本（预处理阶段允许非 Qwen 工具）
import re
import json
from pathlib import Path

import pdfplumber

from . import config


def parse_pdf(pdf_path: Path) -> str:
    """逐页抽取文本 + 表格（表格转为制表符行，避免单元格错位）。"""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            parts.append(f"\n[第{i + 1}页]\n{text}")
            # 表格单独抽取，追加在该页文本后（pdfplumber 表格更保结构）
            for table in page.extract_tables() or []:
                rows = []
                for row in table:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if any(cells):
                        rows.append("\t".join(cells))
                if rows:
                    parts.append("[表格]\n" + "\n".join(rows))
    return "\n".join(parts)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, doc_id: str):
    """条款/章节感知切块：优先在 第X条/第X章/数字标题 处断开，回退定长滑窗。"""
    boundary = re.compile(r"(?=\n(?:第[一二三四五六七八九十百零\d]+[条章节部分]|[一二三四五六七八九十]+、|\d+(?:\.\d+)+\s))")
    sections = boundary.split(text)
    chunks = []
    buf = ""
    for sec in sections:
        if len(buf) + len(sec) <= config.CHUNK_SIZE:
            buf += sec
            continue
        if buf:
            chunks.append(buf)
        # 超长 section 再做定长滑窗
        while len(sec) > config.CHUNK_SIZE:
            chunks.append(sec[: config.CHUNK_SIZE])
            sec = sec[config.CHUNK_SIZE - config.CHUNK_OVERLAP:]
        buf = sec
    if buf.strip():
        chunks.append(buf)
    return [
        {"doc_id": doc_id, "chunk_id": f"{doc_id}#{i}", "text": c.strip()}
        for i, c in enumerate(chunks) if c.strip()
    ]


def iter_domain_docs(domain: str):
    """遍历某领域全部原始文档，yield (doc_id, path)。doc_id = 文件名去扩展名。"""
    root = config.RAW_DIR / domain
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in (".pdf", ".txt") and p.is_file():
            yield p.stem, p


def preprocess_domain(domain: str, force=False):
    """解析领域内全部文档，输出 processed_data/{domain}/{doc_id}.json（chunks 列表）。"""
    out_dir = config.PROCESSED_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for doc_id, path in iter_domain_docs(domain):
        out = out_dir / f"{doc_id}.json"
        if out.exists() and not force:
            continue
        try:
            text = path.read_text(encoding="utf-8") if path.suffix == ".txt" else parse_pdf(path)
        except Exception as e:
            print(f"[WARN] 解析失败 {path}: {e}")
            continue
        chunks = chunk_text(clean_text(text), doc_id)
        out.write_text(json.dumps({"doc_id": doc_id, "source": str(path), "chunks": chunks},
                                  ensure_ascii=False), encoding="utf-8")
        n += 1
        print(f"[{domain}] {doc_id}: {len(chunks)} chunks")
    return n


def load_chunks(domain: str):
    """加载领域内所有已处理文档的 chunks。返回 list[dict]。"""
    out_dir = config.PROCESSED_DIR / domain
    chunks = []
    for f in sorted(out_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        chunks.extend(data["chunks"])
    return chunks
