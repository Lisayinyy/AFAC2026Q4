"""Fine-grained insurance contract segmentation.

Insurance questions are especially sensitive to product identity and clause scope.
This segmenter indexes small, typed facts but keeps the surrounding clause as a
parent evidence group.  It is deterministic and does not use an LLM or embeddings.
"""
from __future__ import annotations

import re

from schema import AtomicFact, EvidenceGroup, SourceRef


_SENTENCE_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")
_ENUM_RE = re.compile(r"(?=(?:^|\s)(?:\([0-9一二三四五六七八九十]+\)|[0-9一二三四五六七八九十]+[、．.]))")
_HEADING_ONLY_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+条|\d+(?:\.\d+){0,3})?\s*"
    r"(?:保险责任|责任免除|其他责任免除|等待期|宽限期|效力中止与恢复|"
    r"保险金计算方法|免赔额|保险期间|犹豫期|合同解除|释义)$"
)

_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("coverage", re.compile(r"保险责任|保险金|给付|保障范围|承担.*责任")),
    ("exclusion", re.compile(r"责任免除|不承担|不予给付|不在保障范围|除外")),
    ("waiting_period", re.compile(r"等待期")),
    ("grace_period", re.compile(r"宽限期")),
    ("contract_state", re.compile(r"效力中止|效力恢复|合同终止|合同解除|复效")),
    ("deductible_formula", re.compile(r"免赔额|计算方法|计算公式|给付比例|×\s*\d+%")),
    ("eligibility", re.compile(r"投保年龄|被保险人|投保人|指定医疗机构")),
    ("definition", re.compile(r"释义|是指|定义")),
]


def classify_insurance_fact(text: str, section: str = "") -> str:
    scope = f"{section} {text}"
    # Specific state/period/formula labels should outrank the generic word 责任.
    priority = [4, 3, 2, 5, 1, 0, 6, 7]
    for index in priority:
        label, pattern = _TYPE_PATTERNS[index]
        if pattern.search(scope):
            return label
    return "general"


def _clean(text: str) -> str:
    text = text.replace("\uf043", " ").replace("\uf076", " ").replace("\uf077", " ")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _atomic_units(text: str) -> list[str]:
    """Split at sentence/list boundaries without severing condition-result clauses."""
    sentences = [item.strip() for item in _SENTENCE_RE.findall(text) if item.strip()]
    units: list[str] = []
    for sentence in sentences:
        # A PDF paragraph often flattens a numbered exclusion list into one sentence.
        parts = [part.strip() for part in _ENUM_RE.split(sentence) if part.strip()]
        units.extend(parts or [sentence])
    return units or ([text] if text else [])


def _evidence_groups(units: list[str], target_chars: int = 480) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    current_type = ""
    for unit in units:
        unit_type = classify_insurance_fact(unit)
        boundary = (
            current
            and (size + len(unit) > target_chars
                 or (unit_type != "general" and current_type not in {"", "general", unit_type}))
        )
        if boundary:
            groups.append(current)
            current, size, current_type = [], 0, ""
        current.append(unit)
        size += len(unit)
        if unit_type != "general":
            current_type = unit_type
    if current:
        groups.append(current)
    return groups


def _document_title(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lead = _clean(chunks[0].get("text", ""))
    match = re.search(r"[^。]{2,80}?(?:保险|险)(?:（[^）]+）)?条款", lead)
    return (match.group(0) if match else lead[:100]).strip()


def segment_insurance_chunks(chunks: list[dict]) -> list[dict]:
    """Convert legacy chunks into atomic index records with parent-group metadata."""
    if not chunks:
        return []
    title = _document_title(chunks)
    records: list[dict] = []
    group_order = 0
    for chunk in chunks:
        raw = _clean(chunk.get("text", ""))
        if not raw:
            continue
        section = _clean(chunk.get("section", "")) or raw[:80]
        source = SourceRef(
            doc_id=str(chunk["doc_id"]),
            domain="insurance",
            chunk_id=chunk["chunk_id"],
            page=chunk.get("page"),
            section=section,
        )
        if chunk.get("is_table"):
            # Table rows are already strong atomic boundaries; retain the whole table
            # as parent so headers remain available after a row is matched.
            lines = [line.strip() for line in chunk["text"].splitlines() if line.strip()]
            grouped_units = [lines] if len(lines) <= 3 else [
                [lines[0], line] for line in lines[1:]
            ]
        else:
            units = _atomic_units(raw)
            grouped_units = _evidence_groups(units)

        for local_group, group_units in enumerate(grouped_units):
            if not group_units:
                continue
            parent_text = (
                "\n".join(group_units) if chunk.get("is_table")
                else "".join(group_units)
            )
            group_id = f"insurance:{source.doc_id}:g{source.chunk_id}.{local_group}"
            fact_type = classify_insurance_fact(parent_text, section)
            parent = EvidenceGroup(
                node_id=group_id,
                document_id=f"insurance:{source.doc_id}",
                section_id=f"insurance:{source.doc_id}:s:{section}",
                title=section,
                text=parent_text,
                order=group_order,
                source=source,
                fact_type=fact_type,
                metadata={"is_table": bool(chunk.get("is_table"))},
            )
            group_order += 1

            facts = group_units if chunk.get("is_table") else _atomic_units(parent_text)
            # Heading-only records are useful as context but too weak to be standalone
            # evidence.  Attach the title to search text and keep one typed atom.
            if _HEADING_ONLY_RE.match(parent_text):
                facts = [parent_text]
            for fact_order, fact_text in enumerate(facts):
                fact_text = fact_text.strip()
                if not fact_text:
                    continue
                fact_id = f"{group_id}:f{fact_order}"
                atom_type = classify_insurance_fact(fact_text, section)
                search_text = " ".join(
                    part for part in [title, section, atom_type, fact_text] if part
                )
                atom = AtomicFact(
                    node_id=fact_id,
                    document_id=parent.document_id,
                    section_id=parent.section_id,
                    group_id=group_id,
                    text=fact_text,
                    order=fact_order,
                    source=source,
                    search_text=search_text,
                    fact_type=atom_type,
                    subject=title,
                )
                parent.fact_ids.append(fact_id)
                records.append(atom.to_index_record(parent))
    return records


def expand_parent_groups(records: list[dict], limit: int | None = None) -> list[dict]:
    """Deduplicate matched atoms and return complete parent evidence groups."""
    expanded: list[dict] = []
    seen: set[str] = set()
    for record in records:
        group_id = record.get("group_id") or record.get("region")
        if group_id in seen:
            continue
        seen.add(group_id)
        value = dict(record)
        value["atomic_text"] = record.get("text", "")
        value["text"] = record.get("parent_text") or record.get("text", "")
        value["chunk_type"] = "evidence_group"
        expanded.append(value)
        if limit is not None and len(expanded) >= limit:
            break
    return expanded
