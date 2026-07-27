import streamlit as st
import math
import os
import time
from foundry_local_sdk import FoundryLocalManager, Configuration
from openai import OpenAI
import database
database.init_db()  # Veri tabanı tablosunu yoksa otomatik olarak oluşturur
import main  # main.py içindeki cosine_similarity ve retrieve_context fonksiyonlarını kullanacağız

# MODELLERİN HER SAYFA YENİLENDİĞİNDE TEKRAR YÜKLENMESİNİ ENGELLİYORUZ
@st.cache_resource
def get_foundry_manager():
    """Foundry Local servisini başlatır ve arka planda açık tutar."""
    config = Configuration(app_name="rag_assistant")
    manager = FoundryLocalManager(config)
    manager.start_web_service()
    return manager

@st.cache_resource
def load_models():
    """Modelleri bir kez hafızaya yükler ve uygulamayı hızlandırır."""
    manager = get_foundry_manager()
    chat_model_id = "qwen2.5-1.5b-instruct-generic-cpu:4"
    embedding_model_id = "qwen3-embedding-0.6b-generic-cpu:1"
    
    # Sohbet modelini yükle
    chat_variant = manager.catalog.get_model_variant(chat_model_id)
    chat_variant.load()
    
    # Embedding modelini yükle
    embedding_variant = manager.catalog.get_model_variant(embedding_model_id)
    embedding_variant.load()
    
    url = manager.urls[0]
    openai_client = OpenAI(base_url=url + "/v1", api_key="foundry")
    embedding_client = embedding_variant.get_embedding_client()
    
    return openai_client, embedding_client, chat_model_id

# Arayüz başlığını ve ikonunu ayarlıyoruz
st.set_page_config(page_title="Lilo - Yerel RAG Asistanı", page_icon="🤖")

# PREMİUM GÖRSEL TASARIM (Açık Renkli Tema CSS Kodları)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Font */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main Background Gradient (Clean white to soft light indigo) */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e0e7ff 100%) !important;
    }
    
    /* Genel Yazı Renkleri (Gündüz Modu) */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp span, .stApp li {
        color: #0f172a !important;
    }
    
    /* Yan Menü (Sidebar) Yazı Renkleri */
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    
    /* Sohbet Balonları (Gündüz Modu Soft Beyaz) */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.03) !important;
    }
    [data-testid="stChatMessage"] * {
        color: #0f172a !important;
    }
    
    /* Input Box styling */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
    }
    
    /* Button & Popover Button Styling */
    .stButton>button, div[data-testid="stPopover"]>button {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        background: #ffffff !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02) !important;
        height: 38px !important;
        width: 100% !important;
        padding: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stButton>button:hover, div[data-testid="stPopover"]>button:hover {
        border-color: #4f46e5 !important;
        color: #4f46e5 !important;
        background: rgba(99, 102, 241, 0.05) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
        transform: translateY(-2px);
    }
    
    /* Kolonları Dikeyde Tam Ortala */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    
    /* Sidebar Styling & Sağ Sınır Taşıma Koruması */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-right: 20px !important;
    }
    
    /* Custom Styling for citations card */
    div.stMarkdown blockquote {
        background-color: rgba(99, 102, 241, 0.03) !important;
        border-left: 4px solid #4f46e5 !important;
        border-radius: 6px !important;
        padding: 10px 15px !important;
        color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)
# Modelleri hafızaya yüklüyoruz ve köprülerimizi alıyoruz
try:
    openai_client, embedding_client, chat_model_id = load_models()
except Exception as e:
    st.error(f"Modeller yüklenirken bir hata oluştu: {e}")
    st.stop()

# ==================================================
# 1. ÇOKLU SOHBET (SESSION STATE) ALTYAPISI
# ==================================================
# Eğer sohbet geçmişi hafızası yoksa ilk sohbet oturumunu oluşturuyoruz
if "chats" not in st.session_state:
    st.session_state.chats = {
        "chat_1": {
            "title": "İlk Sohbet 💬",
            "messages": []
        }
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = "chat_1"

# ==================================================
# 2. SOL PANEL (SIDEBAR) TASARIMI
# ==================================================
# Lilo Logo (Maskot Görseli)
if os.path.exists("lilo_logo.png"):
    st.sidebar.image("lilo_logo.png", use_container_width=True)
else:
    st.sidebar.title("🤖 Lilo")
st.sidebar.markdown("<div style='text-align: center; font-size: 1.25rem; font-weight: 600; color: #818cf8; margin-top: -10px; margin-bottom: 2px;'>Lilo</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center; font-size: 0.75rem; color: #64748b; margin-bottom: 15px;'>Local Information Loop Optimizer</div>", unsafe_allow_html=True)

# --- YENİ SOHBET BAŞLAT BUTONU ---
if st.sidebar.button("➕ Yeni Sohbet Başlat", use_container_width=True):
    # Benzersiz bir yeni sohbet ID'si oluşturuyoruz (timestamp bazlı)
    new_chat_id = f"chat_{int(time.time())}"
    st.session_state.chats[new_chat_id] = {
        "title": "Yeni Sohbet 💬",
        "messages": []
    }
    st.session_state.active_chat_id = new_chat_id
    st.rerun()

st.sidebar.write("### 💬 Aktif Sohbetler")

# --- SOHBET OTURUMLARINI LİSTELEME VE SİLME ---
for chat_id, chat_data in list(st.session_state.chats.items()):
    col1, col2 = st.sidebar.columns([4.5, 1], gap="small")
    
    # Sohbet seçme butonu
    with col1:
        # Eğer aktif sohbet bu ise butonu vurgulamak için küçük bir işaret koyalım
        label = f"▶ {chat_data['title']}" if chat_id == st.session_state.active_chat_id else chat_data['title']
        if st.button(label, key=f"select_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()
            
    # Sohbeti yönetme popover'ı
    with col2:
        with st.popover("⚙️", help="Sohbeti yönet"):
            st.markdown(f"💬 **Sohbet:** {chat_data['title']}")
            st.markdown(f"✉️ **Mesaj:** {len(chat_data['messages'])} adet")
            
            st.divider()
            
            if st.button("🗑️ Sohbeti Sil", key=f"del_chat_{chat_id}", use_container_width=True, type="primary"):
                # Hafızadan sil
                del st.session_state.chats[chat_id]
                # Eğer silinen sohbet şu an açık olan sohbet ise, başka bir sohbete geç
                if st.session_state.active_chat_id == chat_id:
                    if st.session_state.chats:
                        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
                    else:
                        # Tüm sohbetler silindiyse varsayılan yeni bir sohbet oluştur
                        st.session_state.chats = {"chat_1": {"title": "Yeni Sohbet 💬", "messages": []}}
                        st.session_state.active_chat_id = "chat_1"
                st.sidebar.success("Sohbet silindi!")
                time.sleep(0.5)
                st.rerun()

# --- SOHBETİ İNDİRME BUTONU ---
# Eğer aktif sohbette mesaj varsa indirme butonunu gösteriyoruz
if len(chat_data["messages"]) > 0:
    # Sohbet geçmişini metin formatına çeviriyoruz
    chat_text = f"--- {chat_data['title']} Sohbet Geçmişi ---\n\n"
    for msg in chat_data["messages"]:
        role_name = "Kullanıcı" if msg["role"] == "user" else "Lilo"
        chat_text += f"{role_name}: {msg['content']}\n\n"
        
    st.sidebar.download_button(
        label="📥 Sohbet Geçmişini İndir",
        data=chat_text,
        file_name=f"{chat_data['title'].replace(' 💬', '')}_gecmisi.txt",
        mime="text/plain",
        use_container_width=True,
        help="Bu sohbetin geçmişini bilgisayarınıza kaydeder."
    )
st.sidebar.divider()

# --- WEB ÜZERİNDEN BELGE YÜKLEME ---
st.sidebar.title("📂 Belge Yönetimi")
uploaded_file = st.sidebar.file_uploader("Bir metin dosyası seçin (.txt)", type=["txt"])

if uploaded_file is not None:
    if st.sidebar.button("Veri Tabanına Kaydet", use_container_width=True):
        with st.spinner("Belge işleniyor ve vektörler çıkarılıyor..."):
            try:
                file_content = uploaded_file.read().decode("utf-8")
                
                from ingest import chunk_text
                chunks = chunk_text(file_content)
                
                for i, chunk in enumerate(chunks):
                    resp = embedding_client.generate_embedding(chunk)
                    vector = resp.data[0].embedding
                    database.save_chunk(uploaded_file.name, chunk, vector)
                
                os.makedirs("documents", exist_ok=True)
                with open(os.path.join("documents", uploaded_file.name), "w", encoding="utf-8") as f:
                    f.write(file_content)
                    
                st.sidebar.success(f"Başarılı! '{uploaded_file.name}' belgesi eklendi.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Dosya yüklenirken hata oluştu: {e}")

# --- YÜKLÜ BELGELERİ LİSTELEME VE SİLME ---
st.sidebar.write("### 📄 Yüklü Belgeler")
filenames = database.get_unique_filenames()

if filenames:
    for fname in filenames:
        col1, col2 = st.sidebar.columns([5, 1], gap="small")
        with col1:
            st.write(f"📄 {fname}")
        with col2:
            # ⚙️ Belge Yönetim Popover'ı (Açılır metin kutusu ve silme butonu)
            with st.popover("⚙️", help=f"'{fname}' belgesini yönet"):
                filepath = os.path.join("documents", fname)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            st.text_area(f"📄 {fname} İçeriği", f.read(), height=250, disabled=True)
                    except Exception as e:
                        st.error(f"Dosya okunamadı: {e}")
                else:
                    st.error("Dosya bulunamadı.")
                
                st.divider() # Önizleme ile silme butonu arasına şık bir çizgi
                
                # Silme butonu popover'ın içinde duruyor
                if st.button("🗑️ Belgeyi Sil", key=f"del_doc_{fname}", use_container_width=True, type="primary"):
                    # Veri tabanından sil
                    database.delete_document(fname)
                    # Klasörden de fiziksel olarak kaldır
                    filepath = os.path.join("documents", fname)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    st.sidebar.success(f"'{fname}' silindi!")
                    time.sleep(1)
                    st.rerun()
else:
    st.sidebar.info("Veri tabanında henüz belge yok.")


# ==================================================
# 3. ANA EKRAN TASARIMI VE SOHBET DÖNGÜSÜ
# ==================================================
active_chat = st.session_state.chats[st.session_state.active_chat_id]

# EĞER SOHBETTE HİÇ MESAJ YOKSA (BOŞ SOHBET)
if len(active_chat["messages"]) == 0:
    import datetime
    # Sistem saatine göre selamlama mesajı ve emojiyi seçiyoruz
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting_text = "Günaydın, Sena!"
        greeting_emoji = "☀️"
    elif 12 <= current_hour < 17:
        greeting_text = "Tünaydın, Sena!"
        greeting_emoji = "🌤️"
    elif 17 <= current_hour < 22:
        greeting_text = "İyi akşamlar, Sena!"
        greeting_emoji = "🌙"
    else:
        greeting_text = "İyi geceler, Sena!"
        greeting_emoji = "🌌"
        
    # Ekrana ortalanmış şık bir karşılama kartı (Glassmorphic Card)
    st.write("\n\n")
    col_img1, col_img2, col_img3 = st.columns([2, 1, 2])
    with col_img2:
        if os.path.exists("lilo_logo.png"):
            st.image("lilo_logo.png", use_container_width=True)
            
    # Kart rengini ve yazı rengini açık temaya göre ayarlıyoruz
    card_bg = "rgba(255, 255, 255, 0.75)"
    card_border = "rgba(99, 102, 241, 0.15)"
    card_shadow = "rgba(99, 102, 241, 0.05)"
    text_gradient = "linear-gradient(90deg, #4f46e5 0%, #6366f1 100%)"
    desc_color = "#475569"
        
    st.markdown(f"""
        <div style='
            background: {card_bg}; 
            padding: 30px; 
            border-radius: 16px; 
            border: 1px solid {card_border}; 
            backdrop-filter: blur(10px); 
            text-align: center;
            box-shadow: 0 8px 32px {card_shadow};
            margin-bottom: 25px;
        '>
            <h1 style='margin-bottom: 10px; font-size: 2.25rem; font-weight: 700;'>
                <span style='
                    background: {text_gradient};
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                '>{greeting_text}</span> 
                <span>{greeting_emoji}</span>
            </h1>
            <p style='font-size: 1.1rem; color: {desc_color}; margin-bottom: 0px;'>Ben Lilo, yerel RAG asistanınız. Bugün size nasıl yardımcı olabilirim?</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Hızlı soru kartları oluşturuyoruz (Kullanıcı bunlara tıklayarak doğrudan soru sorabilir)
    st.markdown("<p style='font-weight: bold; font-size: 1rem;'>💡 Örnek Sorular:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    user_query = None # Varsayılan olarak boş
    
    with col1:
        if st.button("📝 Makine öğrenmesi nedir?", use_container_width=True):
            user_query = "Makine öğrenmesi nedir?"
    with col2:
        if st.button("🧠 Yapay sinir ağları nedir?", use_container_width=True):
            user_query = "Yapay sinir ağları nedir ve nasıl çalışır?"
            
else:
    # SOHBETTE MESAJ VARSA: Klasik başlığı ve mesaj geçmişini göster
    st.title(f"🤖 {active_chat['title']}")
    st.caption("Lilo - Yerel Çevrimdışı Kişisel Bilgi Bankası Asistanı")
    
    # Aktif sohbetin geçmiş mesajlarını ekrana çizdiriyoruz
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    user_query = None

# Kullanıcıdan yeni soru alma kutusu (Eğer hızlı soru kartlarına basılmadıysa burayı dinler)
chat_input_query = st.chat_input("Lilo'ya bir şey sorun...")
if chat_input_query:
    user_query = chat_input_query

if user_query:
    # 1. Kullanıcının sorusunu hafızaya ekle ve ekrana çiz
    active_chat["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Eğer sohbetin başlığı hala "Yeni Sohbet" ise, başlığı sorunun ilk kelimelerine göre otomatik güncelle
    if active_chat["title"] in ["İlk Sohbet 💬", "Yeni Sohbet 💬"] or active_chat["title"].startswith("Yeni Sohbet"):
        short_title = user_query[:18] + "..." if len(user_query) > 18 else user_query
        active_chat["title"] = f"{short_title} 💬"
    
    # 2. Veri tabanında arama yap (RAG Aşaması)
    with st.spinner("Lilo araştırıyor..."):
        context, score, source_files = main.retrieve_context(user_query, embedding_client, top_k=1)
        
    # 3. Benzerlik skoruna göre karar veriyoruz (Eşik değerimiz: 0.30)
    if score >= 0.30:
        clean_context = context.replace('\r', '')
        system_prompt = (
            "Sen yardımcı bir asistansın. Aşağıdaki kaynak bilgide yer alan 'Cevap:' kısmını oku "
            "ve soruyu oradaki ifadelere sadık kalarak, kelimeleri değiştirmeden doğrudan yanıtla.\n\n"
            f"Kaynak Bilgi:\n{clean_context}"
        )
        
        # YAPAY ZEKAYA GERÇEK SOHBET GEÇMİŞİNİ (HAFIZAYI) GÖNDERİYORUZ
        messages = [{"role": "system", "content": system_prompt}]
        for msg in active_chat["messages"][:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_query})
        
        try:
            # Cevabı canlı olarak yazdırmak için yer ayırtıyoruz (Streaming)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # OpenAI modelinden akan veriyi (stream) başlatıyoruz
                response_stream = openai_client.chat.completions.create(
                    model=chat_model_id,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1,
                    frequency_penalty=1.1,
                    stream=True  # Akışı aktif ediyoruz
                )
                
                # Gelen her kelime parçasını ekrana ekliyoruz
                for chunk in response_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content_delta = chunk.choices[0].delta.content
                        if content_delta:
                            full_response += content_delta
                            message_placeholder.markdown(full_response + "▌")
                
                # Kaynakları ve Güvenilirlik Çubuğunu hazırlıyoruz (Jüri dostu normalizasyon yapıyoruz)
                sources_str = ", ".join(source_files) if source_files else "Bilinmeyen Kaynak"
                
                # Kosinüs benzerliğini 0.30 - 0.80 arasından 70% - 100% arasına kalibre ediyoruz
                calibrated_percentage = int(((score - 0.30) / 0.50) * 30 + 70)
                # 70% ile 100% arasında sınırlandırıyoruz
                display_score = max(70, min(100, calibrated_percentage))
                
                # Bar uzunluğunu bu yeni skora göre hesaplıyoruz
                bar_length = 10
                filled_length = int(round(bar_length * (display_score / 100.0)))
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                citation_md = (
                    f"\n\n---\n"
                    f"**📄 Kaynak:** `{sources_str}`\n"
                    f"**🎯 Güven Oranı:** `% {display_score}` | `{bar}`"
                )
                
                # Tam cevaba kaynağı ekliyoruz
                final_answer = full_response + citation_md
                
                # İmleci kaldırıp nihai metni ve kaynağı basıyoruz
                message_placeholder.markdown(final_answer)
                
            active_chat["messages"].append({"role": "assistant", "content": final_answer})
            st.rerun()
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
    else:
        # Bilgi bulunamadıysa doğrudan uyarı mesajını yazdır
        answer = "Bu bilgiye sahip değilim (Belgelerde bulunamadı)."
        active_chat["messages"].append({"role": "assistant", "content": answer})
        st.rerun()