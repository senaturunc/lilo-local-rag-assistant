import io
import os
import re
from collections import Counter

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
PAGE_RE = re.compile(r"^\s*(\d+\s*/\s*\d+|sayfa\s*\d+|page\s*\d+)\s*$", re.I)


def _is_noise_line(s: str) -> bool:
    """Menü, reklam, sayfa altlığı gibi cümle olmayan kısa satırları tespit eder."""
    if URL_RE.search(s) and len(s) < 160:
        return True
    if PAGE_RE.match(s):
        return True
    if re.fullmatch(r"[\d\W_]+", s):
        return True
    if len(s) < 40 and s.isupper():
        return True
    words = s.split()
    if len(s) < 40 and any(ch.isdigit() for ch in s) and s[-1] not in ".!?:;" and len(words) <= 5:
        return True
    return False


def _drop_noise(pages):
    """Sayfa metinlerinden gürültü satırlarını atıp tek metin döndürür."""
    # Birden fazla sayfada tekrar eden satırlar (başlık/altlık) gürültüdür
    repeated = set()
    if len(pages) >= 2:
        counts = Counter()
        for p in pages:
            for line in {ln.strip() for ln in p.split("\n") if ln.strip()}:
                counts[line] += 1
        limit = max(2, int(len(pages) * 0.4))
        repeated = {line for line, c in counts.items() if c >= limit}

    cleaned_pages = []
    for p in pages:
        kept = []
        for line in p.split("\n"):
            s = line.strip()
            if not s:
                kept.append("")
                continue
            if s in repeated or _is_noise_line(s):
                continue
            kept.append(line)
        cleaned_pages.append("\n".join(kept))
    return "\n\n".join(cleaned_pages)


def clean_pdf_text(text: str) -> str:
    text = text.replace("\u00ad", "")               # gizli tire
    text = re.sub(r"-\n(?=\w)", "", text)           # satır sonu tire birleştirme
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # tek satır sonlarını boşluğa çevir
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(data: bytes) -> str:
    pages = []
    try:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
    except Exception:
        pages = []

    if not "".join(pages).strip():
        # PyMuPDF başarısız olduysa veya boş döndüyse pypdf ile dene
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception:
            pages = []

    text = clean_pdf_text(_drop_noise(pages))
    if not text:
        raise ValueError("PDF'den metin çıkarılamadı (taranmış/görüntü tabanlı PDF olabilir).")
    return text


def extract_text(filename: str, data: bytes) -> str:
    """Dosya uzantısına göre ham metni çıkarır."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        return data.decode("utf-8-sig", errors="replace")

    if ext == ".pdf":
        return _extract_pdf(data)

    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")