import io
import os


def extract_text(filename: str, data: bytes) -> str:
    """Dosya uzantısına göre ham metni çıkarır."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return data.decode("utf-8-sig", errors="replace")

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages)
        if not text.strip():
            raise ValueError("PDF'den metin çıkarılamadı (taranmış/görüntü tabanlı PDF olabilir).")
        return text

    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")