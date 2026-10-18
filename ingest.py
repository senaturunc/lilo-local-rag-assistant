import os
from foundry_local_sdk import FoundryLocalManager, Configuration
import database
from chunking import chunk_text
from loaders import extract_text

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")


def ingest_document(filename, text, embedding_client):
    """Bir belgeyi parçalar, vektörlerini çıkarır ve veri tabanına kaydeder.
    Aynı isimli eski kayıtlar önce silinir. Kaydedilen chunk sayısını döndürür."""
    database.delete_document(filename)
    chunks = chunk_text(text)
    for chunk in chunks:
        resp = embedding_client.generate_embedding(chunk)
        database.save_chunk(filename, chunk, resp.data[0].embedding)
    return len(chunks)


def main():
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()

    embedding_model_id = "qwen3-embedding-0.6b-generic-cpu:1"
    variant = manager.catalog.get_model_variant(embedding_model_id)
    if not variant.is_cached:
        print(f"Embedding modeli ({embedding_model_id}) indiriliyor...")
        variant.download(progress_callback=lambda p: print(f"İndirme: {p:.1f}%", flush=True))
    variant.load()
    client = variant.get_embedding_client()

    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        print(f"Hata: {docs_dir} klasörü bulunamadı.")
        return

    for filename in os.listdir(docs_dir):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
        print(f"İşleniyor: {filename}")
        try:
            with open(os.path.join(docs_dir, filename), "rb") as f:
                text = extract_text(filename, f.read())
            n = ingest_document(filename, text, client)
            print(f"  -> {n} parça kaydedildi.")
        except Exception as e:
            print(f"  -> HATA: {e}")

    print("Tamamlandı.")


if __name__ == "__main__":
    main()