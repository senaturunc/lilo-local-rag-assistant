import streamlit as st
import os
import re
import time
import datetime
from foundry_local_sdk import FoundryLocalManager, Configuration
from openai import OpenAI
import database
import main  # retrieve_chunks(query, embedding_client, top_k) -> [{"score","content","filename"}, ...]
from loaders import extract_text

database.init_db()  # Tablo yoksa otomatik oluşturur

# Eşik değerlendirme verisiyle seçildi (talimatlı sorgu): konu dışı sorular <= 0.41, cevaplılar >= 0.48
SIMILARITY_THRESHOLD = 0.45
TOP_K = 3


# ==================================================
# YARDIMCI FONKSİYONLAR
# ==================================================
def build_citation(source_files, score):
    """Kaynak dosya adı ve ham benzerlik skorundan gösterilecek metni üretir."""
    sources_str = ", ".join(source_files) if source_files else "Bilinmeyen kaynak"
    filled = max(0, min(10, int(round(score * 10))))
    bar = "█" * filled + "░" * (10 - filled)
    return (
        f"\n\n---\n"
        f"**📄 Kaynak:** `{sources_str}`\n"
        f"**🎯 Benzerlik skoru:** `{score:.2f}` | `{bar}`"
    )


def is_looping(text, window=30, min_repeats=3):
    """Metnin son parçası, metin içinde birkaç kez tekrar ediyorsa True döner."""
    if len(text) < window * min_repeats:
        return False
    return text.count(text[-window:]) >= min_repeats


def trim_answer(text, max_sentences=2):
    """Cevabı en fazla max_sentences tam cümleyle sınırlar, yarım kalan cümleyi atar."""
    text = text.strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:max_sentences]).strip()
    if out and out[-1] not in ".!?…":
        cut = max(out.rfind("."), out.rfind("!"), out.rfind("?"))
        if cut > 0:
            out = out[:cut + 1]
    return out or text


def best_sentences(query, results, embedding_client, n=2, max_candidates=36):
    """Getirilen chunk'lardan sorguya en yakın n cümleyi belgeden aynen seçer."""
    per_chunk = []
    for r in results:
        sents = []
        for s in re.split(r"(?<=[.!?])\s+", r["content"].replace("\r", "")):
            s = re.sub(r"https?://\S+|www\.\S+", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) >= 30:
                sents.append(s)
        per_chunk.append(sents)

    # Chunk'lar arasında dönüşümlü seç, hiçbirinin cümleleri tamamen dışarıda kalmasın
    candidates, i = [], 0
    while len(candidates) < max_candidates and any(i < len(c) for c in per_chunk):
        for c in per_chunk:
            if i < len(c):
                candidates.append(c[i])
        i += 1
    candidates = candidates[:max_candidates]
    if not candidates:
        return []

    q_vec = embedding_client.generate_embedding(main.QUERY_INSTRUCT + query).data[0].embedding
    scored = []
    for s in candidates:
        v = embedding_client.generate_embedding(s).data[0].embedding
        scored.append((main.cosine_similarity(q_vec, v), s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]


def render_sources(sources):
    """Kullanılan kaynak parçaları açılır kutuda gösterir."""
    if not sources:
        return
    with st.expander("🔍 Kullanılan kaynak parçalar"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {s['filename']}**  (benzerlik: `{s['score']:.2f}`)")
            snippet = s["content"][:600] + ("…" if len(s["content"]) > 600 else "")
            st.caption(snippet)


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

    chat_variant = manager.catalog.get_model_variant(chat_model_id)
    chat_variant.load()

    embedding_variant = manager.catalog.get_model_variant(embedding_model_id)
    embedding_variant.load()

    url = manager.urls[0]
    openai_client = OpenAI(base_url=url + "/v1", api_key="foundry")
    embedding_client = embedding_variant.get_embedding_client()

    return openai_client, embedding_client, chat_model_id


# ==================================================
# SAYFA AYARLARI VE TASARIM
# ==================================================
st.set_page_config(page_title="Lilo - Yerel RAG Asistanı", page_icon="🤖")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e0e7ff 100%) !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp span, .stApp li {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

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

    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
    }

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

    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-right: 20px !important;
    }

    div.stMarkdown blockquote {
        background-color: rgba(99, 102, 241, 0.03) !important;
        border-left: 4px solid #4f46e5 !important;
        border-radius: 6px !important;
        padding: 10px 15px !important;
        color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Modelleri yükle
try:
    openai_client, embedding_client, chat_model_id = load_models()
except Exception as e:
    st.error(f"Modeller yüklenirken bir hata oluştu: {e}")
    st.stop()

# ==================================================
# 1. ÇOKLU SOHBET (SESSION STATE) ALTYAPISI
# ==================================================
if "chats" not in st.session_state:
    st.session_state.chats = {
        "chat_1": {"title": "İlk Sohbet 💬", "messages": []}
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = "chat_1"

# ==================================================
# 2. SOL PANEL (SIDEBAR)
# ==================================================
if os.path.exists("lilo_logo.png"):
    st.sidebar.image("lilo_logo.png", use_container_width=True)
else:
    st.sidebar.title("🤖 Lilo")
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 1.25rem; font-weight: 600; color: #818cf8; "
    "margin-top: -10px; margin-bottom: 2px;'>Lilo</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 0.75rem; color: #64748b; margin-bottom: 15px;'>"
    "Local Information Loop Optimizer</div>",
    unsafe_allow_html=True,
)

# --- YENİ SOHBET ---
if st.sidebar.button("➕ Yeni Sohbet Başlat", use_container_width=True):
    new_chat_id = f"chat_{int(time.time())}"
    st.session_state.chats[new_chat_id] = {"title": "Yeni Sohbet 💬", "messages": []}
    st.session_state.active_chat_id = new_chat_id
    st.rerun()

st.sidebar.write("### 💬 Aktif Sohbetler")

# --- SOHBET LİSTESİ VE YÖNETİMİ ---
for chat_id, chat_data in list(st.session_state.chats.items()):
    col1, col2 = st.sidebar.columns([4.5, 1], gap="small")

    with col1:
        label = f"▶ {chat_data['title']}" if chat_id == st.session_state.active_chat_id else chat_data["title"]
        if st.button(label, key=f"select_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    with col2:
        with st.popover("⚙️", help="Sohbeti yönet"):
            st.markdown(f"💬 **Sohbet:** {chat_data['title']}")
            st.markdown(f"✉️ **Mesaj:** {len(chat_data['messages'])} adet")
            st.divider()

            if st.button("🗑️ Sohbeti Sil", key=f"del_chat_{chat_id}", use_container_width=True, type="primary"):
                del st.session_state.chats[chat_id]
                if st.session_state.active_chat_id == chat_id:
                    if st.session_state.chats:
                        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
                    else:
                        st.session_state.chats = {"chat_1": {"title": "Yeni Sohbet 💬", "messages": []}}
                        st.session_state.active_chat_id = "chat_1"
                st.sidebar.success("Sohbet silindi!")
                time.sleep(0.5)
                st.rerun()

# --- SOHBETİ İNDİRME (aktif sohbet) ---
current_chat = st.session_state.chats[st.session_state.active_chat_id]
if len(current_chat["messages"]) > 0:
    chat_text = f"--- {current_chat['title']} Sohbet Geçmişi ---\n\n"
    for msg in current_chat["messages"]:
        role_name = "Kullanıcı" if msg["role"] == "user" else "Lilo"
        chat_text += f"{role_name}: {msg['content']}{msg.get('citation', '')}\n\n"

    st.sidebar.download_button(
        label="📥 Sohbet Geçmişini İndir",
        data=chat_text,
        file_name=f"{current_chat['title'].replace(' 💬', '')}_gecmisi.txt",
        mime="text/plain",
        use_container_width=True,
        help="Bu sohbetin geçmişini bilgisayarınıza kaydeder.",
    )
st.sidebar.divider()

# --- GÖRÜNÜM SEÇENEKLERİ (varsayılan: kapalı, eski düzen) ---
quote_only = st.sidebar.checkbox(
    "📄 Sadece belgeden alıntı göster (LLM'siz)",
    value=False,
    help="Açıkken cevap yazılmaz, belgeden en ilgili cümleler aynen gösterilir.",
)
show_sources = st.sidebar.checkbox(
    "🔍 Kaynak parçaları göster",
    value=False,
    help="Cevabın altında kullanılan belge parçalarını açılır kutuda gösterir.",
)
st.sidebar.divider()

# --- BELGE YÜKLEME ---
uploaded_file = st.sidebar.file_uploader(
    "Bir belge seçin (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"]
)

if uploaded_file is not None:
    if st.sidebar.button("Veri Tabanına Kaydet", use_container_width=True):
        with st.spinner("Belge işleniyor ve vektörler çıkarılıyor..."):
            try:
                from ingest import ingest_document

                data = uploaded_file.getvalue()
                text = extract_text(uploaded_file.name, data)
                n = ingest_document(uploaded_file.name, text, embedding_client)

                os.makedirs("documents", exist_ok=True)
                with open(os.path.join("documents", uploaded_file.name), "wb") as f:
                    f.write(data)

                st.sidebar.success(f"'{uploaded_file.name}' eklendi ({n} parça).")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Dosya yüklenirken hata oluştu: {e}")

# --- YÜKLÜ BELGELER ---
st.sidebar.write("### 📄 Yüklü Belgeler")
filenames = database.get_unique_filenames()

if filenames:
    for fname in filenames:
        col1, col2 = st.sidebar.columns([5, 1], gap="small")
        with col1:
            st.write(f"📄 {fname}")
        with col2:
            with st.popover("⚙️", help=f"'{fname}' belgesini yönet"):
                filepath = os.path.join("documents", fname)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            preview = extract_text(fname, f.read())
                        st.text_area(f"📄 {fname} İçeriği", preview, height=250, disabled=True)
                    except Exception as e:
                        st.error(f"Dosya okunamadı: {e}")
                else:
                    st.error("Dosya bulunamadı.")

                st.divider()

                if st.button("🗑️ Belgeyi Sil", key=f"del_doc_{fname}", use_container_width=True, type="primary"):
                    database.delete_document(fname)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    st.sidebar.success(f"'{fname}' silindi!")
                    time.sleep(1)
                    st.rerun()
else:
    st.sidebar.info("Veri tabanında henüz belge yok.")

# ==================================================
# 3. ANA EKRAN VE SOHBET DÖNGÜSÜ
# ==================================================
active_chat = st.session_state.chats[st.session_state.active_chat_id]
user_query = None

if len(active_chat["messages"]) == 0:
    # BOŞ SOHBET: karşılama ekranı
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting_text, greeting_emoji = "Günaydın, Sena!", "☀️"
    elif 12 <= current_hour < 17:
        greeting_text, greeting_emoji = "Tünaydın, Sena!", "🌤️"
    elif 17 <= current_hour < 22:
        greeting_text, greeting_emoji = "İyi akşamlar, Sena!", "🌙"
    else:
        greeting_text, greeting_emoji = "İyi geceler, Sena!", "🌌"

    st.write("\n\n")
    col_img1, col_img2, col_img3 = st.columns([2, 1, 2])
    with col_img2:
        if os.path.exists("lilo_logo.png"):
            st.image("lilo_logo.png", use_container_width=True)

    st.markdown(f"""
        <div style='
            background: rgba(255, 255, 255, 0.75);
            padding: 30px;
            border-radius: 16px;
            border: 1px solid rgba(99, 102, 241, 0.15);
            backdrop-filter: blur(10px);
            text-align: center;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.05);
            margin-bottom: 25px;
        '>
            <h1 style='margin-bottom: 10px; font-size: 2.25rem; font-weight: 700;'>
                <span style='
                    background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                '>{greeting_text}</span>
                <span>{greeting_emoji}</span>
            </h1>
            <p style='font-size: 1.1rem; color: #475569; margin-bottom: 0px;'>
                Ben Lilo, yerel RAG asistanınız. Bugün size nasıl yardımcı olabilirim?
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-weight: bold; font-size: 1rem;'>💡 Örnek Sorular:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Dijital imza nedir?", use_container_width=True):
            user_query = "Dijital imza nedir?"
    with col2:
        if st.button("🔑 Açık anahtarlı şifreleme nasıl çalışır?", use_container_width=True):
            user_query = "Açık anahtarlı şifreleme nasıl çalışır?"
else:
    # MESAJ VARSA: başlık ve geçmiş
    st.title(f"🤖 {active_chat['title']}")
    st.caption("Lilo - Yerel Çevrimdışı Kişisel Bilgi Bankası Asistanı")

    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"] + message.get("citation", ""))
            if show_sources:
                render_sources(message.get("sources"))

chat_input_query = st.chat_input("Lilo'ya bir şey sorun...")
if chat_input_query:
    user_query = chat_input_query

if user_query:
    # 1. Soruyu kaydet ve çiz
    active_chat["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if active_chat["title"].startswith(("İlk Sohbet", "Yeni Sohbet")):
        short_title = user_query[:18] + "..." if len(user_query) > 18 else user_query
        active_chat["title"] = f"{short_title} 💬"

    # 2. Retrieval (talimatlı sorgu, top-3)
    with st.spinner("Lilo araştırıyor..."):
        results = main.retrieve_chunks(user_query, embedding_client, top_k=TOP_K)

    best_score = results[0]["score"] if results else 0.0

    # 3a. Eşik altı: bilgi yok (st.rerun() scripti burada sonlandırır)
    if best_score < SIMILARITY_THRESHOLD:
        active_chat["messages"].append(
            {"role": "assistant", "content": "Bu bilgiye sahip değilim (Belgelerde bulunamadı)."}
        )
        st.rerun()

    # Eşiği geçen parçalar
    good = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
    source_files = sorted({r["filename"] for r in good})
    citation = build_citation(source_files, best_score)
    sources = [
        {"filename": r["filename"], "score": r["score"], "content": r["content"]}
        for r in good
    ]

    top_content = good[0]["content"].replace("\r", "")

    # 3b. Soru/Cevap formatlı belge: LLM'siz, cevabı olduğu gibi çıkar
    if "Cevap:" in top_content:
        m = re.search(r"Cevap:\s*(.+)", top_content, re.DOTALL)
        answer = m.group(1).strip() if m else top_content
        active_chat["messages"].append(
            {"role": "assistant", "content": answer, "citation": citation, "sources": sources}
        )
        st.rerun()

    # 3c. Chunk'lardan en ilgili cümleleri seç (hem alıntı modu hem LLM bunu kullanır)
    with st.spinner("En ilgili ifadeler seçiliyor..."):
        picked = best_sentences(user_query, good, embedding_client, n=3)

    # Alıntı modu (LLM'siz)
    if quote_only:
        if picked:
            answer = "**Belgelerden en ilgili ifadeler:**\n\n" + "\n\n".join(f"> {s}" for s in picked[:2])
        else:
            answer = "Bu bilgiye sahip değilim (Belgelerde bulunamadı)."
        active_chat["messages"].append(
            {"role": "assistant", "content": answer, "citation": citation, "sources": sources}
        )
        st.rerun()

    # 3d. Varsayılan: LLM, seçilen ilgili cümlelere dayanarak kısa cevap yazar
    evidence = "\n".join(f"- {s}" for s in picked) if picked else top_content[:600]

    system_prompt = (
        "You are a question answering assistant. "
        "Answer the question using only the evidence sentences. "
        "Write the answer in Turkish, in one or two short sentences, stating the answer directly. "
        "Do not copy unrelated parts of the evidence and do not add outside information."
    )
    example_user = (
        "Evidence:\n- Dijital imza, dijital belgeleri kriptografik olarak imzalama teknolojisinin ismidir.\n"
        "- Bu hash değerini özel anahtarla şifreler.\n\n"
        "Question: Dijital imza nedir?"
    )
    example_answer = "Dijital imza, dijital belgeleri kriptografik olarak imzalama teknolojisidir."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_answer},
        {"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {user_query}"},
    ]

    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            stream = openai_client.chat.completions.create(
                model=chat_model_id,
                messages=messages,
                max_tokens=120,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    if is_looping(full_response):
                        cut = full_response.find(full_response[-30:])
                        full_response = full_response[:cut].rstrip(" ,;") + " …"
                        break
                    placeholder.markdown(full_response + "▌")

            final_text = trim_answer(full_response, max_sentences=2)
            placeholder.markdown(final_text + citation)

        active_chat["messages"].append(
            {"role": "assistant", "content": final_text, "citation": citation, "sources": sources}
        )
        st.rerun()
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")