import math
from foundry_local_sdk import FoundryLocalManager, Configuration
from openai import OpenAI
import database

# Qwen3-Embedding sorgu tarafında talimatla daha iyi çalışır (belgelere eklenmez).
QUERY_INSTRUCT = "Instruct: Given a question, retrieve the passage that answers it\nQuery: "


def cosine_similarity(v1, v2):
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    return dot_product / (magnitude_v1 * magnitude_v2)


def retrieve_chunks(query, embedding_client, top_k=3):
    """Sorguya en benzer top_k chunk'ı döndürür. Aynı içerikli kopyalar elenir.

    Dönüş: [{"score": float, "content": str, "filename": str}, ...] (skora göre azalan)
    """
    resp = embedding_client.generate_embedding(QUERY_INSTRUCT + query)
    query_vector = resp.data[0].embedding

    chunks = database.get_all_chunks()
    if not chunks:
        return []

    scored = [(cosine_similarity(query_vector, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda x: -x[0])

    results, seen = [], set()
    for score, c in scored:
        key = c["content"].strip()
        if key in seen:
            continue
        seen.add(key)
        results.append({"score": score, "content": c["content"], "filename": c["filename"]})
        if len(results) >= top_k:
            break
    return results


def retrieve_context(query, embedding_client, top_k=1):
    """Eski arayüz (CLI için): (birleşik metin, en iyi skor, dosya adları)."""
    results = retrieve_chunks(query, embedding_client, top_k)
    if not results:
        return "Veri tabanında kayıtlı belge bulunamadı.", 0.0, []
    context = "\n---\n".join(r["content"] for r in results)
    files = sorted({r["filename"] for r in results})
    return context, results[0]["score"], files


def main():
    print("Foundry Local başlatılıyor...")
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()

    url = manager.urls[0]

    chat_model_id = "qwen2.5-1.5b-instruct-generic-cpu:4"
    embedding_model_id = "qwen3-embedding-0.6b-generic-cpu:1"

    print("Modeller hafızaya yükleniyor, lütfen bekleyin...")

    chat_variant = manager.catalog.get_model_variant(chat_model_id)
    if not chat_variant.is_cached:
        print(f"Sohbet modeli ({chat_model_id}) indiriliyor...")
        chat_variant.download(progress_callback=lambda p: print(f"İndirme: {p:.1f}%", flush=True))
    chat_variant.load()

    embedding_variant = manager.catalog.get_model_variant(embedding_model_id)
    embedding_variant.load()

    print("Modeller hazır!")

    embedding_client = embedding_variant.get_embedding_client()
    openai_client = OpenAI(base_url=url + "/v1", api_key="foundry")

    print("\n" + "=" * 50)
    print("Lokal RAG Asistanı Hazır!")
    print("Soru sormaya başlayabilirsiniz (Çıkmak için 'exit' yazın).")
    print("=" * 50 + "\n")

    threshold = 0.45

    while True:
        user_query = input("Soru: ")
        if user_query.strip().lower() == "exit":
            print("Görüşmek üzere!")
            break

        if not user_query.strip():
            continue

        results = retrieve_chunks(user_query, embedding_client, top_k=3)
        if not results or results[0]["score"] < threshold:
            print("\nCevap: Bu bilgiye sahip değilim (Belgelerde bulunamadı).\n" + "-" * 50 + "\n")
            continue

        good = [r for r in results if r["score"] >= threshold]
        context = "\n\n".join(r["content"] for r in good)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using ONLY the sources below. "
                    "If the answer is not in the sources, say you don't know. "
                    "Answer concisely, in Turkish.\n\n"
                    f"Sources:\n{context}"
                ),
            },
            {"role": "user", "content": user_query},
        ]

        try:
            response = openai_client.chat.completions.create(
                model=chat_model_id,
                messages=messages,
                max_tokens=250,
                temperature=0.2,
            )
            print(f"\nCevap: {response.choices[0].message.content}\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Hata oluştu: {e}")


if __name__ == "__main__":
    main()