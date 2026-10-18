from chunking import chunk_text, chunk_general, is_qa_format


def test_qa_format_keeps_question_and_answer_together():
    text = "Soru: A nedir?\n\nCevap: A budur.\n\n\nSoru: B nedir?\n\nCevap: B şudur."
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert "Soru: A nedir?" in chunks[0] and "Cevap: A budur." in chunks[0]


def test_is_qa_format_detects_plain_text_as_false():
    assert not is_qa_format("Bu sıradan bir paragraf. İçinde etiket yok.")


def test_general_chunks_respect_size_and_overlap():
    text = " ".join(f"Bu {i}. cümledir." for i in range(200))
    chunks = chunk_general(text, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)
    # Ardışık chunk'lar arasında örtüşme olmalı
    assert chunks[0].split()[-1] in chunks[1]


def test_long_sentence_is_hard_split():
    text = "a" * 2000
    chunks = chunk_general(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 4