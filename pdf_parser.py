import fitz  # PyMuPDF


def extract_text(uploaded_file) -> str:
    """Extract all text from a PDF uploaded via Streamlit."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_page_count(uploaded_file) -> int:
    """Get the number of pages in a PDF."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count
