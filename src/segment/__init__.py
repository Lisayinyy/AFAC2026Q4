"""Domain-specific document segmenters."""

from .insurance import segment_insurance_chunks
from .financial_report import segment_financial_report_chunks
from .regulatory import segment_regulatory_chunks

__all__ = [
    "segment_financial_report_chunks", "segment_insurance_chunks",
    "segment_regulatory_chunks",
]
