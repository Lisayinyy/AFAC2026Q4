"""Article-aware segmentation for regulations and enforcement documents."""
from __future__ import annotations

import re

from schema import AtomicFact, EvidenceGroup, SourceRef


_ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+条")
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千万0-9]+章")
_ITEM_RE = re.compile(r"^(?:（[一二三四五六七八九十百千万0-9]+）|\([0-9]+\)|[一二三四五六七八九十]+、)")
_SENTENCE_RE = re.compile(r"[^。！？!?；;]+[。！？!?；;]?")


def classify_regulatory_role(text: str, doc_id: str = "") -> str:
    """Distinguish binding rules from case narrative and party arguments."""
    if re.search(r"当事人.{0,20}(?:提出|申辩|辩称)|陈述申辩|听证中.{0,12}(?:提出|认为)", text):
        return "party_argument"
    if re.search(r"经查明|经调查|违法事实|事实如下|本案事实|上述行为", text):
        return "case_fact"
    if re.search(r"我会认为|本局认为|经复核|决定如下|现决定|处罚决定", text):
        return "authority_finding"
    if doc_id.startswith("strict_") or _ARTICLE_RE.search(text):
        return "formal_rule"
    return "formal_rule"


def _clean(text: str) -> str:
    text = re.sub(r"(?:^|\n)[—-]?\s*\d+\s*[—-]?(?=\n|$)", "\n", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title(chunks: list[dict]) -> str:
    lead = " ".join(_clean(c.get("text", "")) for c in chunks[:3])
    match = re.search(r"《[^》]{2,100}》|[^。\n]{2,80}(?:办法|条例|规定|准则|指引|法)", lead)
    return match.group(0) if match else lead[:120]


def _facts_from_part(text: str) -> list[str]:
    clean = _clean(text)
    if not clean:
        return []
    if _ITEM_RE.match(clean):
        return [clean]
    facts = [item.strip() for item in _SENTENCE_RE.findall(clean) if item.strip()]
    return facts or [clean]


def segment_regulatory_chunks(chunks: list[dict]) -> list[dict]:
    """Index paragraphs/items as atoms and expand hits to their complete article."""
    if not chunks:
        return []
    records: list[dict] = []
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(str(chunk["doc_id"]), []).append(chunk)

    for doc_id, doc_chunks in by_doc.items():
        document_title = _title(doc_chunks)
        virtual_chunks: list[dict] = []
        article_boundary = re.compile(
            r"(?m)(?=^第[一二三四五六七八九十百千万0-9]+条)"
        )
        for chunk in doc_chunks:
            text = _clean(chunk.get("text", ""))
            pieces = [piece.strip() for piece in article_boundary.split(text) if piece.strip()]
            for piece in pieces:
                value = dict(chunk)
                value["text"] = piece
                virtual_chunks.append(value)
        chapter = ""
        groups: list[tuple[str, str, list[dict]]] = []
        current_title = ""
        current_chapter = ""
        current_parts: list[dict] = []
        current_is_article = False

        def flush() -> None:
            nonlocal current_title, current_chapter, current_parts, current_is_article
            if current_parts:
                groups.append((current_chapter, current_title or current_chapter or "前言", current_parts))
            current_title, current_chapter, current_parts = "", "", []
            current_is_article = False

        for chunk in virtual_chunks:
            text = _clean(chunk.get("text", ""))
            if not text:
                continue
            first_line = text.splitlines()[0].strip()
            if _CHAPTER_RE.match(first_line):
                flush()
                chapter = first_line[:100]
                continue
            if _ARTICLE_RE.match(first_line):
                flush()
                article = _ARTICLE_RE.match(first_line)
                current_title = article.group(0) if article else first_line[:80]
                current_chapter = chapter
                current_parts = [chunk]
                current_is_article = True
                continue
            if current_parts and current_is_article and _ITEM_RE.match(first_line):
                current_parts.append(chunk)
            else:
                flush()
                current_title = chunk.get("section") or first_line[:80]
                current_chapter = chapter
                current_parts = [chunk]
                current_is_article = False
        flush()

        for group_order, (group_chapter, article_title, parts) in enumerate(groups):
            parent_text = "\n".join(_clean(part.get("text", "")) for part in parts)
            if not parent_text:
                continue
            group_id = f"regulatory:{doc_id}:a{group_order}"
            role = classify_regulatory_role(parent_text, doc_id)
            first = parts[0]
            source_section = " / ".join(x for x in [group_chapter, article_title] if x)
            source = SourceRef(
                doc_id=doc_id, domain="regulatory", chunk_id=first["chunk_id"],
                page=first.get("page"), section=source_section,
            )
            parent = EvidenceGroup(
                node_id=group_id,
                document_id=f"regulatory:{doc_id}",
                section_id=f"regulatory:{doc_id}:s:{group_chapter or 'root'}",
                title=article_title,
                text=parent_text,
                order=group_order,
                source=source,
                fact_type=role,
                metadata={"chapter": group_chapter, "article": article_title, "role": role},
            )
            fact_order = 0
            for part in parts:
                for fact_text in _facts_from_part(part.get("text", "")):
                    fact_role = classify_regulatory_role(fact_text, doc_id)
                    fact_id = f"{group_id}:f{fact_order}"
                    atom = AtomicFact(
                        node_id=fact_id,
                        document_id=parent.document_id,
                        section_id=parent.section_id,
                        group_id=group_id,
                        text=fact_text,
                        order=fact_order,
                        source=SourceRef(
                            doc_id=doc_id, domain="regulatory", chunk_id=part["chunk_id"],
                            page=part.get("page"), section=source_section,
                        ),
                        search_text=" ".join(
                            x for x in [document_title, group_chapter, article_title, fact_role, fact_text] if x
                        ),
                        fact_type=fact_role,
                        subject=document_title,
                        qualifiers={"chapter": group_chapter, "article": article_title, "role": fact_role},
                    )
                    parent.fact_ids.append(fact_id)
                    records.append(atom.to_index_record(parent))
                    fact_order += 1
    return records


def expand_regulatory_groups(records: list[dict], limit: int | None = None) -> list[dict]:
    expanded: list[dict] = []
    seen: set[str] = set()
    for record in records:
        group_id = record.get("group_id")
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
