# 🤖 Lilo - Yerel RAG Yapay Zeka Asistanı

Lilo (Local Information Loop Optimizer), **Microsoft Foundry Local** altyapısını kullanarak tamamen çevrimdışı (internetsiz) ve yerel olarak çalışan bir **RAG (Retrieval-Augmented Generation)** kişisel bilgi bankası asistanıdır. 

Bu proje, ders kuralları, sınav takvimi ve ders notları gibi yüklenen özel dokümanlardan anlamsal arama yaparak kullanıcının sorularını yerel bir yapay zeka modeliyle (Qwen 2.5 1.5B Instruct) kararlı ve güvenilir şekilde yanıtlar.

---

## 🌟 Öne Çıkan Özellikler

* **🔒 %100 Yerel ve Güvenli (Offline):** Hiçbir bulut servisine veya API anahtarına ihtiyaç duymadan, sıfır ağ çağrısıyla tamamen kullanıcının CPU'su üzerinde çalışır.
* **⚡ Canlı Akış (Streaming):** Yapay zeka kelimeleri üretirken eşzamanlı olarak ekrana yazar. ChatGPT benzeri akış deneyimiyle CPU üzerindeki bekleme algısını en aza indirir.
* **📄 Kaynak Gösterimi (Citations):** Yapay zeka cevap verdiği her cümlenin altına hangi dokümandan (Örn: `course_rules.txt`) beslendiğini otomatik olarak ekler.
* **🎯 Güvenilirlik Skoru & Barı (Confidence Meter):** Kosinüs benzerliği (Cosine Similarity) algoritması sonucunu kullanıcı dostu bir yüzdeye ve görsel ilerleme çubuğuna (`% 88 | █████████░`) dönüştürerek jüriye sunum kolaylığı sağlar.
* **👁️ Yan Panel Doküman Önizleme & Silme:** Yüklenen belgeler tarayıcıdan çıkmadan tek tıkla (`👁️` butonu) önizlenebilir veya veri tabanından kalıcı olarak silinebilir (`🗑️`).
* **🎨 Premium Light Theme Tasarımı:** Outfit yazı tipi, buzlu cam efektleri (glassmorphism) ve yumuşak geçişli pastel gradyan renklerle donatılmış fütüristik arayüz.

---

## 🛠️ Sistem Mimarisi

Proje dört temel katmandan oluşmaktadır:
1. **İstemci Arayüzü (Client Layer):** Streamlit tabanlı premium açık tema web arayüzü.
2. **Sunucu/Boru Hattı (RAG Pipeline):** OpenAI uyumlu API köprüsüyle Foundry Local LLM ve Embedding modellerine bağlanma.
3. **Veri Katmanı (Data Layer):** SQLite3 tabanlı vektör veri tabanı. Metin parçalarını ve bunların 600 boyutlu fütüristik gömmelerini (embedding) saklar.
4. **AI Katmanı (AI Layer):** Microsoft Foundry Local üzerinde çalışan yerel modeller:
   * **Embedding Modeli:** `qwen3-embedding-0.6b-generic-cpu:1`
   * **Dil Modeli (LLM):** `qwen2.5-1.5b-instruct-generic-cpu:4`

---

## ⚙️ Kurulum ve Çalıştırma

### Gereksinimler
* Python 3.10 veya üzeri
* Microsoft Foundry Local CLI / Masaüstü Uygulaması

### Adımlar

1. Gerekli Python kütüphanelerini yükleyin:
   ```bash
   pip install streamlit openai foundry-local-sdk
   ```

2. Proje dizinine gidin ve Streamlit uygulamasını başlatın:
   ```bash
   streamlit run app.py
   ```

3. Tarayıcınızda açılan ekranda sol paneldeki dosya yükleme aracını kullanarak `documents/` klasöründeki metin belgelerini sisteme yükleyin ve soru sormaya başlayın!

---

## 📝 Tasarım ve NLP Kararları (Jüri Notları)

* **Halüsinasyon Engelleme (Eşik Değeri):** Kosinüs benzerlik skoru `0.30`'un altında kalan sorularda model çalıştırılmaz; doğrudan *"Bu bilgiye sahip değilim (Belgelerde bulunamadı)"* güvenli yanıtı döner.
* **Sayısal Kararlılık:** Türkçe dil yapısında Qwen modelinin sonsuz virgül ve rakam döngülerine girmesini engellemek için veri tabanındaki tarih ve sayılar yazıya dönüştürülmüştür. Modelin `frequency_penalty` parametresi `1.1` ve `temperature` parametresi `0.1` olarak set edilerek en kararlı cevap üretimi sağlanmıştır.
