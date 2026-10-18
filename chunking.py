import re

_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')


def is_qa_format(text: str) -> bool:
    """Metin 'Soru:' / 'Cevap:' satırları içeriyorsa True döner."""
    return bool(re.search(r'(?m)^\s*Soru:', text)) and bool(re.search(r'(?m)^\s*Cevap:', text))


def chunk_qa(text: str) -> list[str]:
    """Her 'Soru:' satırında yeni blok başlatır; Soru+Cevap aynı chunk'ta kalır."""
    text = text.replace('\r\n', '\n')
    parts = re.split(r'(?m)^(?=\s*Soru:)', text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    sentences = []
    for para in re.split(r'\n\s*\n', text):
        para = re.sub(r'\s+', ' ', para).strip()
        if para:
            sentences.extend(s for s in _SENT_SPLIT.split(para) if s)
    return sentences


def chunk_general(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Cümle sınırlarına saygı duyan, örtüşmeli (overlap) chunk'lama."""
    text = text.replace('\r\n', '\n')

    # chunk_size'dan uzun tek cümleleri zorla böl
    pieces = []
    for s in _split_sentences(text):
        while len(s) > chunk_size:
            pieces.append(s[:chunk_size])
            s = s[chunk_size:]
        pieces.append(s)

    chunks, current, length = [], [], 0
    for s in pieces:
        if current and length + len(s) + 1 > chunk_size:
            chunks.append(' '.join(current))
            # Son cümlelerden overlap kadarını yeni chunk'a taşı
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(s)
        length += len(s) + 1

    if current:
        chunks.append(' '.join(current))
    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Formatı algılayıp uygun chunk'lama stratejisini seçer."""
    if is_qa_format(text):
        return chunk_qa(text)
    return chunk_general(text, chunk_size, overlap)