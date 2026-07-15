"""文档解析与建库。

职责:
1. doc_id -> 文件路径解析(处理各领域命名、大小写扩展名差异)。
2. PDF(PyMuPDF)/ TXT / HTML(bs4)抽取文本 + 表格结构化。
3. 章节感知切块,注入元数据(doc_id, domain, chunk 序号, is_table)。
4. 缓存到 cache/<doc_id>.json,避免重复解析(不计入答题 token)。

关键:财报/保险的数值题依赖表格,PyMuPDF find_tables 把表格抽成
Markdown 保留行列关系,作为独立 chunk,避免拍平成乱序文本。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from config import CACHE_ROOT, DOMAIN_DIRS
from semantic_split import split_semantic

CHUNK_SIZE = 1000          # 目标 chunk 字符数(中文,约等价 ~1800 英文字符)
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 180
CACHE_VERSION = 6

_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节条款]|"
    r"[一二三四五六七八九十]+、|[0-9]{1,2}(?:\.[0-9]+)*[、.]|"
    r"（[一二三四五六七八九十0-9]+）|[A-Z][、.]|"
    r"(?:目录|摘要|风险提示|重大事项|发行条款|财务情况|附录|定义与解释).{0,40})"
)


def resolve_doc_path(domain: str, doc_id: str) -> Path | None:
    """把 doc_id 映射到实际文件路径,兼容大小写扩展名与子目录。"""
    base = DOMAIN_DIRS[domain]

    if domain == "regulatory":
        # 法规正文在 txt/,证监会文件在 html/ 或 attachments/
        candidates = [
            base / "txt" / f"{doc_id}.txt",
            base / "html" / f"{doc_id}.html",
            base / "attachments" / f"{doc_id}.pdf",
        ]
    else:
        candidates = [base / f"{doc_id}.pdf", base / f"{doc_id}.PDF"]

    for c in candidates:
        if c.exists():
            return c

    # 兜底:大小写不敏感地在目录树里找
    stem = doc_id.lower()
    for p in base.rglob("*"):
        if p.is_file() and p.stem.lower() == stem:
            return p
    return None


def _table_to_markdown(rows: list[list]) -> str:
    """把表格行列转成紧凑 Markdown,清理空列。"""
    cleaned = []
    for r in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in r]
        # 去掉整行空
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return ""
    lines = [" | ".join(c for c in row if c is not None) for row in cleaned]
    return "\n".join(lines)


def _extract_pdf(path: Path) -> tuple[str, list[dict]]:
    """返回正文及带页码/邻近标题的表格列表。"""
    doc = fitz.open(path)
    text_parts = []
    tables_md: list[dict] = []
    for page_no, page in enumerate(doc, 1):
        page_text = page.get_text("text")
        text_parts.append(page_text)
        try:
            tabs = page.find_tables()
            for table_no, t in enumerate(tabs.tables):
                md = _table_to_markdown(t.extract())
                if md and len(md) > 20:
                    # The text immediately above a table usually carries its name,
                    # unit and consolidated/parent-company scope. Preserve a small
                    # deterministic window instead of sending the whole page.
                    context = ""
                    try:
                        y0 = float(t.bbox[1])
                        clip = fitz.Rect(
                            0, max(0.0, y0 - 140.0), page.rect.width, y0
                        )
                        nearby = [
                            re.sub(r"\s+", " ", line).strip()
                            for line in page.get_text("text", clip=clip).splitlines()
                            if line.strip()
                        ]
                        context = " ".join(nearby[-4:])[-320:]
                    except Exception:
                        pass
                    tables_md.append({
                        "markdown": md,
                        "page": page_no,
                        "table_no": table_no,
                        "context": context,
                    })
        except Exception:
            pass
    doc.close()
    return "\n".join(text_parts), tables_md


def _extract_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def _extract_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading(paragraph: str) -> bool:
    paragraph = paragraph.strip()
    return bool(paragraph and len(paragraph) <= 100 and _HEADING_RE.match(paragraph))


def chunk_text(text: str, doc_id: str, domain: str) -> list[dict]:
    """按标题、段落和句子边界切块，并保存章节/区域元数据。"""
    text = _clean(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[dict] = []
    buf = ""
    section = ""

    def flush() -> None:
        nonlocal buf
        value = buf.strip()
        if not value:
            return
        chunks.append({
            "doc_id": doc_id,
            "domain": domain,
            "chunk_id": len(chunks),
            "text": value,
            "is_table": False,
            "section": section,
            "region": section or "__preamble__",
            "chunk_type": "prose",
            "char_count": len(value),
        })

    for para in paragraphs:
        if _is_heading(para):
            flush()
            section = para
            buf = para + "\n"
            continue
        for sentence_group in split_semantic(para, max_sentences=8, threshold=0.08):
            if len(buf) + len(sentence_group) + 1 <= CHUNK_SIZE:
                buf += sentence_group + "\n"
                continue
            if len(buf.strip()) >= MIN_CHUNK_SIZE:
                previous = buf
                flush()
                tail = previous[-CHUNK_OVERLAP:]
                buf = (section + "\n" if section else "") + tail + sentence_group + "\n"
            else:
                buf += sentence_group + "\n"
    flush()
    return chunks


def _chunk_tables(tables_md: list[dict | str], doc_id: str, domain: str,
                  start_id: int) -> list[dict]:
    """每个表格作为独立 chunk；超长表格按行切分并继承表头。"""
    out = []
    cid = start_id
    last_year_header = ""
    last_header_page: int | None = None
    for table_index, table in enumerate(tables_md):
        if isinstance(table, dict):
            md = table.get("markdown", "")
            page = table.get("page")
            context = table.get("context", "")
            table_no = table.get("table_no", table_index)
        else:  # 兼容旧调用与单测
            md, page, context, table_no = table, None, "", table_index
        raw_lines = [line for line in md.splitlines() if line.strip()]
        has_year_header = any(
            len(re.findall(r"(?:19|20)\d{2}", line)) >= 2
            for line in raw_lines[:3]
        )
        continuation_metric = bool(
            raw_lines and re.search(
                r"营业收入|净利润|现金流量净额|每股收益|净资产收益率|"
                r"总资产|净资产", raw_lines[0]
            )
        )
        if (
            not has_year_header and continuation_metric and last_year_header
            and (page is None or last_header_page is None or page - last_header_page <= 1)
        ):
            md = last_year_header + "\n" + md
            raw_lines.insert(0, last_year_header)
            has_year_header = True

        if has_year_header:
            for line_no, line in enumerate(raw_lines[:3]):
                if len(re.findall(r"(?:19|20)\d{2}", line)) >= 2:
                    header_lines = [line]
                    if line_no + 1 < len(raw_lines):
                        next_line = raw_lines[line_no + 1]
                        numeric_cells = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", next_line)
                        if not numeric_cells and re.search(r"金额|占比|比例|增减", next_line):
                            header_lines.append(next_line)
                    last_year_header = "\n".join(header_lines)
                    last_header_page = page
                    break
        if len(md) <= CHUNK_SIZE * 1.5:
            pieces = [md]
        else:  # 超长表格按行切
            lines = md.split("\n")
            # 跨块重复表头，防止命中后只剩数字而年份/单位丢失。
            header_count = 1
            if len(lines) >= 2:
                next_numbers = re.findall(
                    r"[-+]?\d[\d,]*(?:\.\d+)?%?", lines[1]
                )
                if not next_numbers and re.search(r"金额|占比|比例|增减", lines[1]):
                    header_count = 2
            header = "\n".join(lines[:header_count]).strip()
            pieces, buf = [], header + "\n"
            for ln in lines[header_count:]:
                if len(buf) + len(ln) + 1 > CHUNK_SIZE:
                    pieces.append(buf.strip())
                    buf = header + "\n"
                buf += ln + "\n"
            if buf.strip():
                pieces.append(buf.strip())
        for piece_no, p in enumerate(pieces):
            out.append({"doc_id": doc_id, "domain": domain, "chunk_id": cid,
                        "text": "[表格]\n" + p, "is_table": True,
                        "section": context or "table",
                        "region": f"table:{table_no}",
                        "table_id": f"{doc_id}:p{page or 0}:t{table_no}",
                        "table_piece": piece_no,
                        "table_context": context,
                        "page": page,
                        "chunk_type": "table", "char_count": len(p) + 5})
            cid += 1
    return out


def build_doc(domain: str, doc_id: str, use_cache: bool = True) -> list[dict]:
    """解析单个文档为分块列表(正文 + 表格),带缓存。"""
    CACHE_ROOT.mkdir(exist_ok=True)
    cache_file = CACHE_ROOT / f"{domain}__{doc_id}.json"

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("version") == CACHE_VERSION:
            return cached.get("chunks", [])

    path = resolve_doc_path(domain, doc_id)
    if path is None:
        print(f"[WARN] 未找到文档: domain={domain} doc_id={doc_id}")
        return []

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw, tables_md = _extract_pdf(path)
        chunks = chunk_text(raw, doc_id, domain)
        chunks += _chunk_tables(tables_md, doc_id, domain, len(chunks))
    elif suffix in (".html", ".htm"):
        chunks = chunk_text(_extract_html(path), doc_id, domain)
    else:
        chunks = chunk_text(_extract_text_file(path), doc_id, domain)

    cache_file.write_text(
        json.dumps({"version": CACHE_VERSION, "chunks": chunks}, ensure_ascii=False),
        encoding="utf-8",
    )
    return chunks


if __name__ == "__main__":
    for dom, did in [("financial_reports", "annual_byd_2024_report"),
                     ("regulatory", "strict_v3_017_中华人民共和国反洗钱法")]:
        cs = build_doc(dom, did, use_cache=False)
        n_tab = sum(1 for c in cs if c.get("is_table"))
        total = sum(len(c["text"]) for c in cs)
        print(f"{dom}/{did}: {len(cs)} chunks ({n_tab} 表格), {total} chars")
