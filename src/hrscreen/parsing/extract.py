"""Turn an uploaded resume file (PDF, DOCX or TXT) into plain text."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md")


def _pdf_text(data: bytes) -> str:
    # pdfplumber gives better reading order for multi-column layouts; pypdf is the fallback.
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines = [p.text for p in doc.paragraphs]
    # Resumes often keep skills in tables, so include table cells too.
    for table in doc.tables:
        for row in table.rows:
            lines.extend(cell.text for cell in row.cells)
    return "\n".join(lines)


def extract_text(data: bytes, filename: str) -> str:
    """Extract text from the raw bytes of a resume file; the file extension picks the reader."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix == ".docx":
        return _docx_text(data)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported resume format '{suffix}'. Use one of {SUPPORTED_SUFFIXES}.")


def extract_text_from_path(path: str | Path) -> str:
    path = Path(path)
    return extract_text(path.read_bytes(), path.name)
