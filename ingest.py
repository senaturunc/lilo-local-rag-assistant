import os
import re
from foundry_local_sdk import FoundryLocalManager, Configuration
import database

def chunk_text(text):
    """Metni 'Soru:' ile başlayan bloklara göre ayırır.
    Her blok, o soruya ait 'Cevap:' kısmıyla BİRLİKTE tek chunk olur."""
    text = text.replace('\r\n', '\n')
    
    # "Soru:" kelimesinin göründüğü her yerden yeni bir blok başlat
    # (?=Soru:) -> Soru: kelimesini blok içinde tutarak ondan önce böl
    parts = re.split(r'(?=Soru:)', text)
    
    # Boş olmayan, temizlenmiş parçaları döndür
    return [part.strip() for part in parts if part.strip()]

def main():
    # Foundry Local yöneticisini başlatıyoruz
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()

    # Kullanacağımız yerel kelime vektörü (embedding) modeli
    embedding_model_id = "qwen3-embedding-0.6b-generic-cpu:1"
    
    variant = manager.catalog.get_model_variant(embedding_model_id)
    if not variant.is_cached:
        print(f"Embedding modeli ({embedding_model_id}) indiriliyor, lütfen bekleyin...")
        # İndirme ilerlemesini ekrana yazdırıyoruz
        variant.download(progress_callback=lambda p: print(f"İndirme İlerlemesi: {p:.1f}%", flush=True))
        print("Embedding modeli başarıyla indirildi!")

    print("Embedding modeli hafızaya yükleniyor...")
    variant.load()
    print("Model hazır!")

    client = variant.get_embedding_client()

    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        print(f"Hata: {docs_dir} klasörü bulunamadı.")
        return

    print("\nBelgeler taranıyor ve veri tabanına kaydediliyor...")
    
    # Klasördeki tüm dosyaları oku
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            print(f"\nİşlenen dosya: {filename}")
            
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Dosya içeriğini küçük parçalara ayır
            chunks = chunk_text(text)
            print(f"Toplam {len(chunks)} parçaya ayrıldı.")
            
            for i, chunk in enumerate(chunks):
                print(f"  Parça {i+1} için vektör oluşturuluyor...")
                # Yerel model ile metin parçasının vektörünü (embedding) hesapla
                resp = client.generate_embedding(chunk)
                vector = resp.data[0].embedding
                
                # Veri tabanına kaydet
                database.save_chunk(filename, chunk, vector)
                print(f"  Parça {i+1} veri tabanına kaydedildi.")

    print("\nTüm belgeler başarıyla işlendi ve veri tabanına kaydedildi!")

if __name__ == "__main__":
    main()
