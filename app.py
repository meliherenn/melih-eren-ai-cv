import streamlit as st
from openai import OpenAI
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Melih Eren | Portfolyo", page_icon="👨‍💻", layout="wide")

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """Loads the vector database for RAG."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Allow dangerous deserialization because we trust our own local file
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

# Initial loading with a custom spinner placeholder if needed, or silent
try:
    data = load_data()
    load_css()
    with st.spinner("Sistem Hazırlanıyor..."): # Custom message instead of code trace
        vectorstore = load_rag_engine()
except FileNotFoundError:
    st.error("Gerekli dosyalar (data.json, style.css veya faiss_index) bulunamadı!")
    st.info("Lütfen önce 'build_vector_db.py' dosyasını çalıştırın.")
    st.stop()

# --- API SETUP ---
try:
    API_KEY = st.secrets["CEREBRAS_API_KEY"]
except:
    API_KEY = "csk-m4t4cwj3n9rjnnr2f9trenmvp6jy2ev4p8y896x4wj55jtwf"

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=API_KEY
)

# --- SESSION STATE ---
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "Türkçe"
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# --- HELPER FUNCTIONS ---
def get_system_prompt(lang_data, lang_code, context=""):
    """Constructs the system prompt dynamically from JSON data and RAG context."""
    prompts = lang_data["prompts"]
    experiences = "\n".join([f"{i+1}. {exp}" for i, exp in enumerate(lang_data["experience"])])
    
    cert_list = []
    for i, cert in enumerate(lang_data["certificates"]):
        cert_list.append(f"{i+1}. [{cert['name']}]({cert['url']})")
    certificates = "\n".join(cert_list)
    
    projects = "\n".join([f"{i+1}. {proj}" for i, proj in enumerate(lang_data["projects"])])
    skills = ", ".join([f"{k} ({v})" for k, v in lang_data["skills"].items()])
    style_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(prompts["style_rules"])])

    context_str = ""
    if context:
        context_str = f"\n--- EK BİLGİ (PDF'ten) ---\nKullanıcının sorusuyla ilgili şu detaylar bulundu:\n{context}\nBu bilgileri cevabında kullanabilirsin.\n"

    if lang_code == "tr":
        return f"""
        GÖREV: Sen Melih Eren'in asistanısın. Türkçe konuş.
        --- KİMLİK ---
        {prompts['identity_a']}
        --- ÜSLUP KURALLARI ---
        {style_rules}
        --- DENEYİMLER ---
        {experiences}
        --- SERTİFİKALAR ---
        {certificates}
        --- PROJELER ---
        {projects}
        --- KARİYER HEDEFLERİ ---
        {prompts['career_goals']}
        --- YETENEKLER ---
        {skills}
        {context_str}
        """
    else:
         return f"""
        ROLE: You are Melih Eren's AI assistant. Speak ONLY ENGLISH.
        
        --- IDENTITY ---
        {prompts['identity_a']}
        
        --- STYLE RULES (VERY IMPORTANT) ---
        {style_rules}
        
        --- INSTRUCTIONS ---
        1. ANSWER ONLY THE SPECIFIC QUESTION ASKED. Do not summarize the whole CV unless asked "Tell me about yourself".
        2. If asked about "Projects", talk ONLY about Projects.
        3. If asked about "Certificates", list them using the LINKS provided.
        4. BE CONCISE.
        
        --- EXPERIENCE ---
        {experiences}
        
        --- CERTIFICATES (MANDATORY FORMAT: [Name](Link)) ---
        {certificates}
        
        --- PROJECTS ---
        {projects}
        
        --- CAREER GOALS ---
        {prompts['career_goals']}
        
        --- SKILLS ---
        {skills}
        {context_str}
        
        IMPORTANT: 
        - When listing certificates, YOU MUST USE THE MARKDOWN LINK FORMAT: [Certificate Name](URL). 
        - DO NOT invent roles. For HSD, use the description provided above (Core Team Member, contribution role).
        """

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists(data["profile"]["image"]):
        st.image(data["profile"]["image"], width=150)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    
    st.title(data["profile"]["name"])
    
    language = st.radio("Dil / Language:", ("Türkçe", "English"))
    
    if language == "Türkçe":
        st.caption(data["profile"]["title_tr"])
        lang_key = "tr"
    else:
        st.caption(data["profile"]["title_en"])
        lang_key = "en"
    
    if language != st.session_state.current_lang:
        st.session_state.current_lang = language
        st.session_state.messages = [] 
        st.rerun() 
    
    st.divider()
    
    # PDF DOWNLOAD
    pdf_path = "MELİH EREN.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download CV (PDF)",
                data=pdf_file,
                file_name="Melih_Eren_CV.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    st.markdown("🔗 **Links:**")
    st.markdown(f"[GitHub]({data['profile']['contact']['github']})")
    st.markdown(f"[LinkedIn]({data['profile']['contact']['linkedin']})")
    st.markdown(f"[📧 Email]({data['profile']['contact']['email']})")
    
    st.divider()

    # ADMIN LOGIN
    with st.expander("🔐 Admin Panel"):
        if not st.session_state.is_admin:
            password = st.text_input("Şifre / Password", type="password")
            if st.button("Giriş / Login"):
                # Try to get password from secrets, fallback to a safe default or hardcoded ONLY for local dev
                try:
                    admin_pass = st.secrets["ADMIN_PASSWORD"]
                except:
                    admin_pass = "admin123" # Fallback for local testing only
                
                if password == admin_pass:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Hatalı şifre!")
        else:
            if st.button("Çıkış / Logout"):
                st.session_state.is_admin = False
                st.rerun()

# --- MAIN CONTENT ---
current_data = data[lang_key]
ui_text = current_data["ui"]

st.title("🎓 " + ("Melih Eren | Portfolyo" if lang_key == "tr" else "Melih Eren | Portfolio"))

# --- ADMIN VS USER VIEW ---
if st.session_state.is_admin:
    tab1, tab2 = st.tabs(["🤖 Chat Bot", "⚙️ Admin Panel"])
else:
    tab1 = st.container()
    tab2 = None

with tab1:
    st.write(ui_text["welcome_msg"])
    
    # BUTTONS GRID
    cols = st.columns(3)
    buttons = ui_text["buttons"]
    hidden_prompts = ui_text["hidden_prompts"]
    selected_prompt = None

    for i, btn_label in enumerate(buttons):
        col_index = i % 3
        with cols[col_index]:
            if st.button(btn_label, use_container_width=True):
                selected_prompt = hidden_prompts[i]

    st.divider()

    # CHAT HISTORY CONTAINER
    # We use a container for chat history to keep it organized above the input
    chat_container = st.container()
    
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": ui_text["welcome_msg"]}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # INPUT HANDLING
    user_input = None
    display_text = None

    if prompt := st.chat_input("..."):
        user_input = prompt
        display_text = prompt
    elif selected_prompt:
        user_input = selected_prompt
        index = hidden_prompts.index(selected_prompt)
        display_text = buttons[index]

    if user_input:
        st.session_state.messages.append({"role": "user", "content": display_text})
        # Rerun to immediately show user message, then process response
        # Actually in Streamlit, appending to state and then rerunning is a pattern, 
        # or just displaying it. Let's stick to display to avoid full reload flicker.
        with chat_container:
            with st.chat_message("user"):
                st.markdown(display_text)

        # --- RAG SEARCH ---
        found_context = ""
        if vectorstore:
            try:
                # Search for 2 most relevant chunks
                results = vectorstore.similarity_search(user_input, k=2)
                found_context = "\n".join([doc.page_content for doc in results])
            except Exception as e:
                print(f"Search Error: {e}")

        SYSTEM_PROMPT = get_system_prompt(current_data, lang_key, context=found_context)
        
        # Add explicit instruction to avoid repetition
        if lang_key == "tr":
            SYSTEM_PROMPT += "\nÖNEMLİ: Cevabın kısa, net ve tekrarsız olsun. Kullanıcı aynı soruyu sorsa bile sadece en güncel ve net bilgiyi ver."
        else:
            SYSTEM_PROMPT += "\nIMPORTANT: Be concise and avoid repetition. Even if asked the same question, provide only the most relevant and clear answer."

        try:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Limit history to last 4 messages to prevent context pollution loop
            # We skip the Welcome message (index 0) and take the last 4
            recent_history = st.session_state.messages[1:][-4:] 
            
            for msg in recent_history:
                api_messages.append(msg)
            
            # If came from button, ensure the question is actually sent to the API!
            if selected_prompt:
                 # Case 1: History was added, swap the last display message with hidden prompt
                 if api_messages and api_messages[-1]["role"] == "user":
                     api_messages[-1] = {"role": "user", "content": user_input}
                 # Case 2: History was empty (or only system msg), so we MUST append the user prompt manually
                 else:
                     api_messages.append({"role": "user", "content": user_input})
            
            # Additional safeguard: If user typed manually but history trimming removed it (unlikely but possible)
            elif user_input and (not api_messages or api_messages[-1]["role"] != "user"):
                 api_messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..." if lang_key == "en" else "Düşünüyor..."):
                chat = client.chat.completions.create(
                    model="llama3.1-8b", 
                    messages=api_messages,
                    temperature=0.3
                )
                response = chat.choices[0].message.content
                
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Hata/Error: {e}")

# --- ADMIN PANEL CONTENT ---
if st.session_state.is_admin and tab2:
    with tab2:
        st.header("Veri Düzenleme / Edit Data")
        
        # Edit Profile
        with st.expander("Profil Bilgileri / Profile Info"):
            new_name = st.text_input("İsim", data["profile"]["name"])
            new_title_tr = st.text_input("Ünvan (TR)", data["profile"]["title_tr"])
            new_image = st.text_input("Resim Yolu", data["profile"]["image"])
            if st.button("Profili Kaydet"):
                data["profile"]["name"] = new_name
                data["profile"]["title_tr"] = new_title_tr
                data["profile"]["image"] = new_image
                save_data(data)
                st.success("Kaydedildi!")

        # Edit Experience
        with st.expander("Deneyimler (TR)"):
            exp_text = st.text_area("Her satıra bir deneyim yazın", "\n".join(data["tr"]["experience"]), height=200)
            if st.button("Deneyimleri Kaydet"):
                data["tr"]["experience"] = exp_text.split("\n")
                save_data(data)
                st.success("Kaydedildi!")

        # Edit Skills
        with st.expander("Yetenekler / Skills"):
            st.info("Format: Yetenek (Puan/10)")
            # Simple Dictionary Editor
            skills_input = st.text_area("JSON Formatında Girin", json.dumps(data["tr"]["skills"], indent=4, ensure_ascii=False))
            if st.button("Yetenekleri Kaydet"):
                try:
                    data["tr"]["skills"] = json.loads(skills_input)
                    save_data(data)
                    st.success("Kaydedildi!")
                except Exception as e:
                    st.error(f"JSON Hatası: {e}")

