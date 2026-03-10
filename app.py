import streamlit as st
from openai import OpenAI
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Melih Eren | AI Portfolio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """Loads the vector database for RAG."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        print(f"RAG Load Error: {e}")
        return None

def load_data():
    with open('data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_css():
    with open('style.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- INITIAL LOADING ---
try:
    data = load_data()
    load_css()
    with st.spinner("🚀 Sistem Hazırlanıyor / System Loading..."):
        vectorstore = load_rag_engine()
except FileNotFoundError:
    st.error("❌ Gerekli dosyalar (data.json, style.css veya faiss_index) bulunamadı!")
    st.info("💡 Lütfen önce `python build_vector_db.py` komutunu çalıştırın.")
    st.stop()

# --- API SETUP ---
try:
    API_KEY = st.secrets["CEREBRAS_API_KEY"]
except Exception:
    st.error("🔑 API Key bulunamadı! Lütfen `.streamlit/secrets.toml` dosyasını kontrol edin.")
    st.stop()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=API_KEY
)

# --- SESSION STATE ---
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "Türkçe"
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HELPER FUNCTIONS ---
def get_system_prompt(lang_data, lang_code, context=""):
    """Constructs the system prompt dynamically from JSON data and RAG context."""
    prompts = lang_data["prompts"]
    experiences = "\n".join([f"- {exp}" for exp in lang_data["experience"]])

    cert_list = []
    for cert in lang_data["certificates"]:
        cert_list.append(f"- [{cert['name']}]({cert['url']})")
    certificates = "\n".join(cert_list)

    projects = "\n".join([f"- {proj}" for proj in lang_data["projects"]])
    skills = ", ".join([f"{k}: {v}" for k, v in lang_data["skills"].items()])
    style_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(prompts["style_rules"])])

    context_str = ""
    if context:
        if lang_code == "tr":
            context_str = f"\n--- EK BİLGİ (CV PDF'inden) ---\nKullanıcının sorusuyla ilgili şu detaylar bulundu:\n{context}\nBu bilgileri cevabında doğal şekilde kullan.\n"
        else:
            context_str = f"\n--- ADDITIONAL CONTEXT (from CV PDF) ---\nThe following details were found related to the user's question:\n{context}\nUse this information naturally in your answer.\n"

    if lang_code == "tr":
        return f"""GÖREV: Sen Melih Eren'in kişisel AI asistanısın. Her zaman Türkçe konuş.

--- KİMLİK ---
{prompts['identity_a']}

--- ÜSLUP KURALLARI ---
{style_rules}

--- DENEYİMLER ---
{experiences}

--- SERTİFİKALAR (Her zaman [İsim](Link) formatında göster) ---
{certificates}

--- PROJELER ---
{projects}

--- KARİYER HEDEFLERİ ---
{prompts['career_goals']}

--- YETENEKLER ---
{skills}
{context_str}
ÖNEMLİ KURALLAR:
- Cevabın kısa, net ve tekrarsız olsun.
- SADECE sorulan konuyu cevapla.
- Sertifikaları listelerken HER ZAMAN [Sertifika Adı](URL) formatını kullan.
- HSD için sadece "Core Team Member" unvanını kullan, başka rol uydurma.
- Emoji kullanarak cevapları daha okunaklı yap.
"""
    else:
        return f"""ROLE: You are Melih Eren's personal AI assistant. Always speak ONLY in English.

--- IDENTITY ---
{prompts['identity_a']}

--- STYLE RULES ---
{style_rules}

--- EXPERIENCE ---
{experiences}

--- CERTIFICATES (MANDATORY: Always use [Name](URL) format) ---
{certificates}

--- PROJECTS ---
{projects}

--- CAREER GOALS ---
{prompts['career_goals']}

--- SKILLS ---
{skills}
{context_str}
IMPORTANT RULES:
- Be concise and avoid repetition.
- Answer ONLY the specific question asked.
- When listing certificates, YOU MUST use [Certificate Name](URL) markdown format.
- For HSD, use only "Core Team Member" title, do NOT invent other roles.
- Use emojis to make answers more readable.
"""

# --- SIDEBAR ---
with st.sidebar:
    # Profile Section
    st.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
    if os.path.exists(data["profile"]["image"]):
        st.image(data["profile"]["image"], width=150)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)

    st.title(data["profile"]["name"])

    language = st.radio("🌐 Dil / Language:", ("Türkçe", "English"), horizontal=True)

    if language == "Türkçe":
        st.caption(f"📚 {data['profile']['title_tr']}")
        lang_key = "tr"
    else:
        st.caption(f"📚 {data['profile']['title_en']}")
        lang_key = "en"

    # Language change handler
    if language != st.session_state.current_lang:
        st.session_state.current_lang = language
        st.session_state.messages = []
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # PDF DOWNLOAD - Language-aware
    pdf_key = "cv_pdf_tr" if lang_key == "tr" else "cv_pdf_en"
    pdf_path = data["profile"].get(pdf_key, "")

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            btn_label = "📄 CV İndir (PDF)" if lang_key == "tr" else "📄 Download CV (PDF)"
            st.download_button(
                label=btn_label,
                data=pdf_file,
                file_name=pdf_path,
                mime="application/pdf",
                use_container_width=True
            )

    st.divider()

    # Social Links
    link_label = "🔗 **Bağlantılar:**" if lang_key == "tr" else "🔗 **Links:**"
    st.markdown(link_label)
    st.markdown(f"<a href='{data['profile']['contact']['github']}' target='_blank'>🐙 GitHub</a>", unsafe_allow_html=True)
    st.markdown(f"<a href='{data['profile']['contact']['linkedin']}' target='_blank'>💼 LinkedIn</a>", unsafe_allow_html=True)
    st.markdown(f"<a href='{data['profile']['contact']['email']}'>📧 Email</a>", unsafe_allow_html=True)

    st.divider()

    # Footer
    st.markdown(
        "<div style='text-align:center; opacity: 0.5; font-size: 0.75rem;'>"
        "Built with ❤️ using Streamlit & Cerebras AI"
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ADMIN LOGIN
    with st.expander("🔐 Admin Panel"):
        if not st.session_state.is_admin:
            password = st.text_input(
                "Şifre / Password" if lang_key == "tr" else "Password",
                type="password"
            )
            if st.button("🔓 Giriş / Login" if lang_key == "tr" else "🔓 Login"):
                try:
                    admin_pass = st.secrets["ADMIN_PASSWORD"]
                    if password == admin_pass:
                        st.session_state.is_admin = True
                        st.rerun()
                    else:
                        st.error("❌ Hatalı şifre!" if lang_key == "tr" else "❌ Wrong password!")
                except KeyError:
                    st.error("Admin şifresi secrets içinde tanımlanmamış!" if lang_key == "tr" else "Admin password not configured in secrets!")
        else:
            st.success("✅ Admin girişi aktif" if lang_key == "tr" else "✅ Logged in as Admin")
            if st.button("🚪 Çıkış / Logout" if lang_key == "tr" else "🚪 Logout"):
                st.session_state.is_admin = False
                st.rerun()


# --- MAIN CONTENT ---
current_data = data[lang_key]
ui_text = current_data["ui"]

# Title
title_text = "Melih Eren | Portfolyo" if lang_key == "tr" else "Melih Eren | Portfolio"
st.title(f"🚀 {title_text}")

# --- ADMIN VS USER VIEW ---
if st.session_state.is_admin:
    tab1, tab2 = st.tabs(["🤖 Chat Bot", "⚙️ Admin Panel"])
else:
    tab1 = st.container()
    tab2 = None

with tab1:
    st.markdown(
        f"<p style='text-align:center; font-size:1.1rem; opacity:0.8; margin-bottom:1.5rem;'>{ui_text['welcome_msg']}</p>",
        unsafe_allow_html=True
    )

    # QUICK ACTION BUTTONS
    cols = st.columns(3)
    buttons = ui_text["buttons"]
    hidden_prompts = ui_text["hidden_prompts"]
    selected_prompt = None

    for i, btn_label in enumerate(buttons):
        col_index = i % 3
        with cols[col_index]:
            if st.button(btn_label, use_container_width=True, key=f"btn_{i}"):
                selected_prompt = hidden_prompts[i]

    st.divider()

    # CHAT HISTORY
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.session_state.messages = [{"role": "assistant", "content": ui_text["welcome_msg"]}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # INPUT HANDLING
    user_input = None
    display_text = None

    placeholder_text = "Bir soru sorun..." if lang_key == "tr" else "Ask a question..."
    if prompt := st.chat_input(placeholder_text):
        user_input = prompt
        display_text = prompt
    elif selected_prompt:
        user_input = selected_prompt
        index = hidden_prompts.index(selected_prompt)
        display_text = buttons[index]

    if user_input:
        st.session_state.messages.append({"role": "user", "content": display_text})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(display_text)

        # --- RAG SEARCH ---
        found_context = ""
        if vectorstore:
            try:
                results = vectorstore.similarity_search(user_input, k=3)
                found_context = "\n".join([doc.page_content for doc in results])
            except Exception as e:
                print(f"RAG Search Error: {e}")

        SYSTEM_PROMPT = get_system_prompt(current_data, lang_key, context=found_context)

        try:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Limit history to last 6 messages to keep context relevant
            recent_history = st.session_state.messages[1:][-6:]

            for msg in recent_history:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            # If from button, swap display text with actual prompt for the API
            if selected_prompt:
                if api_messages and api_messages[-1]["role"] == "user":
                    api_messages[-1] = {"role": "user", "content": user_input}
                else:
                    api_messages.append({"role": "user", "content": user_input})
            elif user_input and (not api_messages or api_messages[-1]["role"] != "user"):
                api_messages.append({"role": "user", "content": user_input})

            spinner_text = "🧠 Düşünüyor..." if lang_key == "tr" else "🧠 Thinking..."
            with st.spinner(spinner_text):
                chat = client.chat.completions.create(
                    model="llama-3.3-70b",
                    messages=api_messages,
                    temperature=0.3,
                    max_tokens=1024
                )
                response = chat.choices[0].message.content

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"⚠️ Bir hata oluştu: {e}" if lang_key == "tr" else f"⚠️ An error occurred: {e}"
            st.error(error_msg)


# --- ADMIN PANEL CONTENT ---
if st.session_state.is_admin and tab2:
    with tab2:
        st.header("⚙️ Veri Düzenleme / Edit Data")

        admin_lang = st.radio(
            "Düzenlenecek Dil / Language to Edit:",
            ("Türkçe (tr)", "English (en)"),
            horizontal=True
        )
        edit_lang = "tr" if "tr" in admin_lang else "en"

        st.divider()

        # Edit Profile
        with st.expander("👤 Profil Bilgileri / Profile Info", expanded=False):
            new_name = st.text_input("İsim / Name", data["profile"]["name"])
            new_title_tr = st.text_input("Ünvan (TR)", data["profile"]["title_tr"])
            new_title_en = st.text_input("Title (EN)", data["profile"]["title_en"])
            new_image = st.text_input("Resim Yolu / Image Path", data["profile"]["image"])
            new_github = st.text_input("GitHub URL", data["profile"]["contact"]["github"])
            new_linkedin = st.text_input("LinkedIn URL", data["profile"]["contact"]["linkedin"])
            new_email = st.text_input("Email", data["profile"]["contact"]["email"])

            if st.button("💾 Profili Kaydet / Save Profile", key="save_profile"):
                data["profile"]["name"] = new_name
                data["profile"]["title_tr"] = new_title_tr
                data["profile"]["title_en"] = new_title_en
                data["profile"]["image"] = new_image
                data["profile"]["contact"]["github"] = new_github
                data["profile"]["contact"]["linkedin"] = new_linkedin
                data["profile"]["contact"]["email"] = new_email
                save_data(data)
                st.success("✅ Profil kaydedildi!")

        # Edit Identity
        with st.expander(f"🪪 Kimlik ({edit_lang.upper()}) / Identity", expanded=False):
            new_identity = st.text_area(
                "Kimlik Tanımı / Identity Description",
                data[edit_lang]["prompts"]["identity_a"],
                height=120
            )
            new_career = st.text_area(
                "Kariyer Hedefleri / Career Goals",
                data[edit_lang]["prompts"]["career_goals"],
                height=100
            )
            if st.button(f"💾 Kimliği Kaydet / Save Identity ({edit_lang.upper()})", key=f"save_identity_{edit_lang}"):
                data[edit_lang]["prompts"]["identity_a"] = new_identity
                data[edit_lang]["prompts"]["career_goals"] = new_career
                save_data(data)
                st.success("✅ Kaydedildi!")

        # Edit Experience
        with st.expander(f"💼 Deneyimler ({edit_lang.upper()}) / Experience", expanded=False):
            exp_text = st.text_area(
                "Her satıra bir deneyim / One experience per line",
                "\n".join(data[edit_lang]["experience"]),
                height=250
            )
            if st.button(f"💾 Deneyimleri Kaydet ({edit_lang.upper()})", key=f"save_exp_{edit_lang}"):
                data[edit_lang]["experience"] = [e.strip() for e in exp_text.split("\n") if e.strip()]
                save_data(data)
                st.success("✅ Kaydedildi!")

        # Edit Projects
        with st.expander(f"🚀 Projeler ({edit_lang.upper()}) / Projects", expanded=False):
            proj_text = st.text_area(
                "Her satıra bir proje / One project per line",
                "\n".join(data[edit_lang]["projects"]),
                height=200
            )
            if st.button(f"💾 Projeleri Kaydet ({edit_lang.upper()})", key=f"save_proj_{edit_lang}"):
                data[edit_lang]["projects"] = [p.strip() for p in proj_text.split("\n") if p.strip()]
                save_data(data)
                st.success("✅ Kaydedildi!")

        # Edit Skills
        with st.expander(f"💻 Yetenekler ({edit_lang.upper()}) / Skills", expanded=False):
            st.info("JSON formatında düzenleyin / Edit in JSON format")
            skills_input = st.text_area(
                "Skills JSON",
                json.dumps(data[edit_lang]["skills"], indent=4, ensure_ascii=False),
                height=200
            )
            if st.button(f"💾 Yetenekleri Kaydet ({edit_lang.upper()})", key=f"save_skills_{edit_lang}"):
                try:
                    data[edit_lang]["skills"] = json.loads(skills_input)
                    save_data(data)
                    st.success("✅ Kaydedildi!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON Hatası: {e}")

        # Edit Certificates
        with st.expander(f"📜 Sertifikalar ({edit_lang.upper()}) / Certificates", expanded=False):
            st.info("JSON formatında düzenleyin / Edit in JSON format")
            certs_input = st.text_area(
                "Certificates JSON",
                json.dumps(data[edit_lang]["certificates"], indent=4, ensure_ascii=False),
                height=300
            )
            if st.button(f"💾 Sertifikaları Kaydet ({edit_lang.upper()})", key=f"save_certs_{edit_lang}"):
                try:
                    data[edit_lang]["certificates"] = json.loads(certs_input)
                    save_data(data)
                    st.success("✅ Kaydedildi!")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON Hatası: {e}")

        st.divider()

        # Danger Zone
        with st.expander("⚠️ Tehlikeli Alan / Danger Zone", expanded=False):
            st.warning("Bu işlemler geri alınamaz! / These actions are irreversible!")
            if st.button("🗑️ Sohbet Geçmişini Temizle / Clear Chat History", key="clear_chat"):
                st.session_state.messages = []
                st.success("✅ Sohbet geçmişi temizlendi!")
                st.rerun()
