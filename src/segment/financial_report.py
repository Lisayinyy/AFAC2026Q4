"""Structured financial-report table segmentation.

Each metric row becomes one retrieval atom.  The atom carries table context, header,
years, unit and scope so a value cannot be detached from its reporting period.  On a
hit, reasoning receives the compact parent package ``context + header + metric row``.
"""
from __future__ import annotations

import re
from typing import Any

from schema import AtomicFact, EvidenceGroup, SourceRef


_YEAR_RE = re.compile(r"(?:19|20)\d{2}(?:年|年度|Q[1-4]|[一二三四]季度)?")
_UNIT_RE = re.compile(
    r"(?:单位|金额单位)\s*[：:]?\s*(人民币)?\s*"
    r"(亿元|万元|千元|百万元|元|美元|港元|%|个百分点|股|万元/吨)"
)
_INLINE_UNIT_RE = re.compile(
    r"[（(](亿元|万元|千元|百万元|元|美元|港元|%|个百分点|元/股|元/每股)[）)]"
)
_SCOPE_RE = re.compile(r"合并口径|母公司口径|合并报表|母公司|本集团|本公司")
_NUMBER_RE = re.compile(r"[-+]?\(?\d[\d,，]*(?:\.\d+)?\)?%?")
_FINANCIAL_RE = re.compile(
    r"主要会计数据|财务指标|营业收入|营业总收入|净利润|利润总额|现金流|"
    r"研发投入|研发费用|总资产|净资产|负债|所有者权益|每股收益|净资产收益率|"
    r"营业成本|毛利率|分红|股利|股份回购|资本化|税费|应收账款|存货"
)


def _cells(line: str) -> list[str]:
    cells = [re.sub(r"\s+", " ", cell).strip() for cell in line.split("|")]
    while cells and not cells[0]:
        cells.pop(0)
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _clean_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line).strip()


def _table_lines(text: str) -> list[str]:
    return [
        _clean_line(line)
        for line in text.splitlines()
        if _clean_line(line) and _clean_line(line) != "[表格]"
    ]


def _header_end(lines: list[str]) -> int:
    """Return the final header-line index (inclusive)."""
    for index, line in enumerate(lines[:6]):
        years = _YEAR_RE.findall(line)
        if len(years) >= 1 and ("项目" in line or "科目" in line or "指标" in line or "|" in line):
            # Multi-level headers often put “金额/占比” on the next line.
            if index + 1 < len(lines):
                next_cells = [cell for cell in _cells(lines[index + 1]) if cell]
                if next_cells and not any(_NUMBER_RE.fullmatch(cell) for cell in next_cells):
                    return index + 1
            return index
    first_cells = _cells(lines[0]) if lines else []
    if len(first_cells) >= 2 and not _NUMBER_RE.fullmatch(first_cells[0]) and any(
        _NUMBER_RE.search(cell) for cell in first_cells[1:]
    ):
        return -1
    # Most extracted tables use one title/unit line followed by one column header.
    return min(1, len(lines) - 1)


def _unit(text: str) -> str:
    match = _UNIT_RE.search(text)
    if match:
        return match.group(2)
    inline = _INLINE_UNIT_RE.search(text)
    return inline.group(1) if inline else ""


def _scope(text: str) -> str:
    match = _SCOPE_RE.search(text)
    return match.group(0) if match else ""


def _row_qualifiers(header_lines: list[str], row: str, context: str) -> dict[str, Any]:
    header = " ".join(header_lines)
    years = _YEAR_RE.findall(header)
    row_cells = _cells(row)
    header_cells = _cells(header_lines[-1]) if header_lines else []
    values = [cell for cell in row_cells[1:] if _NUMBER_RE.search(cell)]
    mapping: dict[str, str] = {}
    header_labels = [cell for cell in header_cells if cell]
    row_values = [cell for cell in row_cells[1:] if cell]
    if header_labels and len(header_labels) == len(row_values):
        mapping = dict(zip(header_labels, row_values))
    elif len(header_cells) == len(row_cells):
        for column, value in zip(header_cells[1:], row_cells[1:]):
            if value and (_YEAR_RE.search(column) or _NUMBER_RE.search(value)):
                mapping[column] = value
    elif years and len(values) >= len(years):
        mapping = dict(zip(years, values[:len(years)]))
    return {
        "metric": row_cells[0] if row_cells else row[:80],
        "years": years,
        "unit": _unit(f"{context} {header} {row}"),
        "scope": _scope(f"{context} {header}"),
        "values": mapping,
    }


def _legacy_record(chunk: dict) -> dict:
    value = dict(chunk)
    value.setdefault("search_text", value.get("text", ""))
    value.setdefault("fact_type", "financial_prose")
    return value


def segment_financial_report_chunks(chunks: list[dict]) -> list[dict]:
    """Create row-level index records while preserving prose chunks unchanged."""
    records: list[dict] = []
    for chunk in chunks:
        if not chunk.get("is_table"):
            records.append(_legacy_record(chunk))
            continue
        lines = _table_lines(chunk.get("text", ""))
        if len(lines) < 2:
            records.append(_legacy_record(chunk))
            continue
        end = _header_end(lines)
        header_lines = lines[:end + 1] if end >= 0 else []
        data_lines = lines[end + 1:]
        context = _clean_line(
            chunk.get("table_context") or chunk.get("section") or "财务报表"
        )
        if not data_lines:
            data_lines = [lines[-1]]
            header_lines = lines[:-1]

        for row_order, row in enumerate(data_lines):
            cells = _cells(row)
            # Blank/separator rows have no retrieval value.
            if not any(cells) or not any(_NUMBER_RE.search(cell) for cell in cells[1:]):
                continue
            qualifiers = _row_qualifiers(header_lines, row, context)
            metric = qualifiers["metric"]
            if (
                not metric or _NUMBER_RE.fullmatch(metric)
                or not re.search(r"[\u4e00-\u9fffA-Za-z]", metric)
                or not _FINANCIAL_RE.search(f"{context} {' '.join(header_lines)} {row}")
            ):
                continue
            group_id = f"financial_reports:{chunk['doc_id']}:t{chunk['chunk_id']}:r{row_order}"
            source = SourceRef(
                doc_id=str(chunk["doc_id"]), domain="financial_reports",
                chunk_id=chunk["chunk_id"], page=chunk.get("page"),
                section=context,
            )
            parent_text = "\n".join(
                part for part in [context, *header_lines, row] if part
            )
            parent = EvidenceGroup(
                node_id=group_id,
                document_id=f"financial_reports:{chunk['doc_id']}",
                section_id=f"financial_reports:{chunk['doc_id']}:s:{context}",
                title=context,
                text=parent_text,
                order=row_order,
                source=source,
                fact_type="financial_metric",
                metadata={"is_table": True, **qualifiers},
            )
            search_text = " ".join(
                str(part) for part in [
                    context, " ".join(header_lines), metric,
                    qualifiers["unit"], qualifiers["scope"], row,
                ] if part
            )
            atom = AtomicFact(
                node_id=f"{group_id}:f0",
                document_id=parent.document_id,
                section_id=parent.section_id,
                group_id=group_id,
                text=row,
                order=0,
                source=source,
                search_text=search_text,
                fact_type="financial_metric",
                subject=str(chunk["doc_id"]),
                qualifiers=qualifiers,
            )
            parent.fact_ids.append(atom.node_id)
            records.append(atom.to_index_record(parent))
    return records


def expand_financial_groups(records: list[dict], limit: int | None = None) -> list[dict]:
    expanded: list[dict] = []
    seen: set[str] = set()
    for record in records:
        group_id = record.get("group_id")
        if not group_id:
            expanded.append(record)
        elif group_id not in seen:
            seen.add(group_id)
            value = dict(record)
            value["atomic_text"] = record.get("text", "")
            value["text"] = record.get("parent_text") or record.get("text", "")
            value["chunk_type"] = "evidence_group"
            expanded.append(value)
        if limit is not None and len(expanded) >= limit:
            break
    return expanded
