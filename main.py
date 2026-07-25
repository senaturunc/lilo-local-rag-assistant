import math
from foundry_local_sdk import FoundryLocalManager, Configuration
from openai import OpenAI
import database

def cosine_similarity(v1, v2):
    """İki vektör (sayı listesi) arasındaki benzerliği ölçer."""
    # İki listenin karşılıklı elemanlarını çarpıp topluyoruz
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Vektörlerin büyüklüklerini (uzunluklarını) hesaplıyoruz
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
        
    # Kosinüs benzerliği formülü: çarpım / (büyüklüklerin çarpımı)
    return dot_product / (magnitude_v1 * magnitude_v2)

def retrieve_context(query, embedding_client, top_k=1):
    """Kullanıcının sorusuna en yakın metin parçasını veri tabanından bulur."""
    # 1. Kullanıcının sorduğu sorunun vektörünü (sayı listesini) çıkar
    resp = embedding_client.generate_embedding(query)
    query_vector = resp.data[0].embedding
    
    # 2. Veri tabanındaki (rag.db) tüm kayıtları çek
    chunks = database.get_all_chunks()
    if not chunks:
        return "Veri tabanında kayıtlı belge bulunamadı.", 0.0, []
        
    # 3. Her veri tabanı parçasıyla sorunun kosinüs benzerliğini hesapla
    scored_chunks = []
    for chunk in chunks:
        similarity = cosine_similarity(query_vector, chunk["embedding"])
        scored_chunks.append((similarity, chunk["content"], chunk["filename"]))
        
    # 4. Benzerlik oranına göre en yüksekten en düşüğe doğru sırala
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # 5. En yüksek benzerliğe sahip ilk 'top_k' adet parçayı seç
    best_matches = scored_chunks[:top_k]
    
    # Seçilen parçaları aralarına çizgi koyarak birleştir
    context = "\n---\n".join([content for score, content, filename in best_matches])
    
    # Skor, metin ve kaynak dosyaları geri döndür
    best_score = best_matches[0][0]
    source_files = list(set([filename for score, content, filename in best_matches]))
    return context, best_score, source_files

def main():
    print("Foundry Local başlatılıyor...")
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()
    
    url = manager.urls[0]
    
    # Kullanacağımız modellerin kimlikleri (ID'leri)
    chat_model_id = "qwen2.5-1.5b-instruct-generic-cpu:4"
    embedding_model_id = "qwen3-embedding-0.6b-generic-cpu:1"
    
    # Her iki modeli de hafızaya (RAM) yüklüyoruz
    print("Modeller hafızaya yükleniyor, lütfen bekleyin...")
    
    chat_variant = manager.catalog.get_model_variant(chat_model_id)
    if not chat_variant.is_cached:
        print(f"Sohbet modeli ({chat_model_id}) önbellekte bulunamadı. İndiriliyor, lütfen bekleyin...")
        chat_variant.download(progress_callback=lambda p: print(f"Sohbet Modeli İndirme İlerleme: {p:.1f}%", flush=True))
        print("Sohbet modeli indirme tamamlandı!")
    chat_variant.load()
    
    embedding_variant = manager.catalog.get_model_variant(embedding_model_id)
    embedding_variant.load()
    
    print("Modeller hazır!")
    
    # Köprülerimizi (istemcileri) oluşturuyoruz
    embedding_client = embedding_variant.get_embedding_client()
    openai_client = OpenAI(base_url=url + "/v1", api_key="foundry")
    
    print("\n" + "="*50)
    print("Lokal RAG Asistanı Hazır!")
    print("Soru sormaya başlayabilirsiniz (Çıkmak için 'exit' yazın).")
    print("="*50 + "\n")
    
        # Sürekli çalışan soru-cevap döngüsü
    while True:
        user_query = input("Soru: ")
        if user_query.strip().lower() == "exit":
            print("Görüşmek üzere!")
            break
            
        if not user_query.strip():
            continue
            
        # 1. ADIM: Sorumuzu alıp veri tabanında aratıyoruz, metni ve benzerlik skorunu alıyoruz
        context, score, source_files = retrieve_context(user_query, embedding_client, top_k=1)
        
        # 2. ADIM: Benzerlik skoruna göre karar veriyoruz (Eşik Değerimiz: 0.30)
        if score >= 0.30:
            # Benzerlik yüksek: Bilgiyi yapay zekaya gönderip düzgün cevap ürettiriyoruz
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "Sen ders kurallarını yanıtlayan yardımcı bir asistansın. "
                        "Aşağıdaki kaynak bilgiye dayanarak soruyu yanıtla. "
                        "Kaynaktaki bilgilerin dışına çıkma ve olabildiğince net, kısa cevap ver."
                        f"\n\nKaynak Bilgi:\n{context}"
                    )
                },
                {"role": "user", "content": user_query}
            ]
            
            # 3. ADIM: Yapay zekaya soruyu ve bilgiyi gönderip cevabı alıyoruz
            try:
                response = openai_client.chat.completions.create(
                    model=chat_model_id,
                    messages=messages,
                    max_tokens=150,
                    temperature=0.1
                )
                print(f"\nCevap: {response.choices[0].message.content}\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"Hata oluştu: {e}")
        else:
            # Benzerlik düşük: PDF'teki zorunlu kural gereği doğrudan hata mesajı dönüyoruz
            print("\nCevap: Bu bilgiye sahip değilim (Belgelerde bulunamadı).\n" + "-"*50 + "\n")

# Programın çalışmasını tetikleyen standart Python komutu
if __name__ == "__main__":
    main()