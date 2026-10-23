import pytest

from hrscreen.data import load_resume_folder
from hrscreen.parsing.extract import extract_text, extract_text_from_path

RESUME = "Priya Sharma. Data scientist with 4 years of experience in python, pandas and machine learning."


def make_docx(path, with_table=True):
    from docx import Document

    doc = Document()
    doc.add_paragraph(RESUME)
    if with_table:
        table = doc.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Skills"
        table.rows[0].cells[1].text = "tableau, power bi"
    doc.save(path)


def make_pdf(path):
    fitz = pytest.importorskip("fitz")  # PyMuPDF, only used here to create a test PDF
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), RESUME)
    pdf.save(path)
    pdf.close()


def test_docx_text_includes_table_cells(tmp_path):
    path = tmp_path / "cv.docx"
    make_docx(path)
    text = extract_text_from_path(path)
    assert "machine learning" in text and "power bi" in text


def test_pdf_text_is_extracted(tmp_path):
    path = tmp_path / "cv.pdf"
    make_pdf(path)
    assert "machine learning" in extract_text_from_path(path)


def test_pdf_falls_back_to_pypdf_without_pdfplumber(tmp_path, monkeypatch):
    path = tmp_path / "cv.pdf"
    make_pdf(path)
    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", None)  # makes `import pdfplumber` raise ImportError
    assert "machine learning" in extract_text_from_path(path)


def test_plain_text_and_unknown_formats():
    assert extract_text(b"hello world", "a.txt") == "hello world"
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(b"x", "a.exe")


def test_folder_loader_reads_mixed_formats_and_ignores_others(tmp_path):
    make_docx(tmp_path / "b.docx", with_table=False)
    (tmp_path / "a.txt").write_text(RESUME, encoding="utf-8")
    (tmp_path / "notes.csv").write_text("ignored", encoding="utf-8")
    df = load_resume_folder(tmp_path)
    assert list(df["resume_id"]) == ["a", "b"]


def test_empty_folder_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_folder(tmp_path)
