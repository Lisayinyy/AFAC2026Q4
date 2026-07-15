"""Hierarchical evidence schema used by domain segmenters and retrieval.

The runtime still exposes dictionaries to the existing Agent so this module can be
introduced without rewriting the judging pipeline.  The hierarchy is explicit:

``DocumentNode -> SectionNode -> EvidenceGroup -> AtomicFact``.

Retrieval is performed against :class:`AtomicFact`; reasoning receives its parent
``EvidenceGroup`` together with the matched atom and a stable source reference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceRef:
    doc_id: str
    domain: str
    chunk_id: int | str
    page: int | None = None
    section: str = ""


@dataclass
class DocumentNode:
    node_id: str
    doc_id: str
    domain: str
    title: str = ""
    section_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionNode:
    node_id: str
    document_id: str
    title: str
    order: int
    source: SourceRef
    group_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceGroup:
    node_id: str
    document_id: str
    section_id: str
    title: str
    text: str
    order: int
    source: SourceRef
    fact_type: str = "general"
    fact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicFact:
    node_id: str
    document_id: str
    section_id: str
    group_id: str
    text: str
    order: int
    source: SourceRef
    search_text: str = ""
    fact_type: str = "general"
    subject: str = ""
    qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_index_record(self, parent: EvidenceGroup) -> dict[str, Any]:
        """Return a backwards-compatible retrieval record.

        ``text`` remains the small atomic unit while ``parent_text`` is carried for
        query-time expansion.  Existing callers can continue to use ``doc_id``,
        ``chunk_id``, ``section`` and ``region``.
        """
        return {
            "doc_id": self.source.doc_id,
            "domain": self.source.domain,
            "chunk_id": self.node_id,
            "source_chunk_id": self.source.chunk_id,
            "page": self.source.page,
            "section": self.source.section or parent.title,
            "section_id": self.section_id,
            "region": self.group_id,
            "group_id": self.group_id,
            "fact_id": self.node_id,
            "fact_type": self.fact_type,
            "text": self.text,
            "search_text": self.search_text or self.text,
            "parent_text": parent.text,
            "parent_title": parent.title,
            "is_table": parent.metadata.get("is_table", False),
            "chunk_type": "atomic_fact",
            "char_count": len(self.text),
            "subject": self.subject,
            "qualifiers": self.qualifiers,
        }


def node_to_dict(node: object) -> dict[str, Any]:
    """Serialize schema nodes for cache/debug manifests."""
    return asdict(node)
