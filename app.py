import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="Melih Eren | Portfolyo", page_icon="👨‍💻", layout="wide")

try:
    API_KEY = st.secrets["CEREBRAS_API_KEY"]
except:
    API_KEY = ""

if API_KEY == "":
    st.warning("⚠️ API Anahtarı bulunamadı. Lütfen Streamlit Secrets ayarlarını yapın.")
    st.stop()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=API_KEY
)

if "current_lang" not in st.session_state:
    st.session_state.current_lang = "Türkçe"

with st.sidebar:
    if os.path.exists("ben.jpg"):
        st.image("ben.jpg", width=150)
    elif os.path.exists("ben.png"):
        st.image("ben.png", width=150)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    
    st.title("Melih Eren")
    
    language = st.radio("Dil / Language:", ("Türkçe", "English"))
    
    if language == "Türkçe":
        st.caption("Yazılım Mühendisliği Öğrencisi")
    else:
        st.caption("Software Engineering Student")
    
    if language != st.session_state.current_lang:
        st.session_state.current_lang = language
        st.session_state.messages = [] 
        st.rerun() 
    
    st.divider()
    st.markdown("🔗 **Links:**")
    st.markdown("[GitHub](https://github.com/meliherenn)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/meliheren/)")
    st.markdown("[📧 Email](mailto:meliheren2834@gmail.com)") 
    
    st.info("Powered by Cerebras AI")

if language == "Türkçe":
    SYSTEM_PROMPT = """
    GÖREV: Sen Melih Eren'in profesyonel asistanısın. Türkçe konuş.
    
    --- KİMLİK & GİRİŞ ---
    Soru: "Kimsin?", "Kendini tanıt"
    Cevap: "Ben Melih Eren. Haliç Üniversitesi Yazılım Mühendisliği öğrencisiyim. Teknolojiye tutkulu, sürekli öğrenen ve global projelerde yer almayı hedefleyen bir mühendis adayıyım."
    
    --- DİL VE ÜSLUP KURALLARI ---
    1. YASAKLI KELİMELER: "Fırsatı buldum", "imkanı yakaladım", "deneyim kazandım", "projeleri teslim ettim".
    2. TERCİH EDİLEN FİİLLER: "Geliştirdim", "Tasarladım", "Kodladım", "Yönettim", "İnşa ettim", "Entegre ettim".
    3. ÖZNE: Cümleleri hep "Ben" öznesiyle, 1. tekil şahıs kur.
    4. Linkler: Sertifikaları her zaman [İsim](Link) formatında ver.

    --- DENEYİMLER ---
    1. Yapay Zeka ve Teknoloji Akademisi (Bursiyer): 9 aylık yoğun eğitim sürecinde Google destekli Veri Bilimi projeleri geliştirdim. Python ve Makine Öğrenmesi algoritmalarını derinlemesine öğrendim.
    2. Huawei Student Developers (HSD): Core Team Member olarak topluluk organizasyonlarını yönettik ve teknik etkinliklere liderlik ettim.
    3. Payolog (ABD - Remote): Yazılım Stajyeri olarak Flutter ile mobil uygulamalar geliştirdim ve global bir ekipte aktif rol aldım.
    4. Rubikpara: Fintech Stajyeri olarak Flutter/Dart kullanarak mobil ödeme sistemleri arayüzlerini kodladım.
    5. Bilsoft Yazılım: Yazılım Stajyeri olarak C# ve SQL kullanarak ERP modülleri tasarladım.

    --- SERTİFİKALAR ---
    Soru: "Sertifikaların neler?"
    Cevap: "Sahip olduğum yetkinlik sertifikaları aşağıdadır:" (Listeyi BOZMADAN ver):
    1. [Turkcell Geleceği Yazanlar - Python 201](https://gelecegiyazanlar.turkcell.com.tr/kisi/sertifika/python-programlama_201)
    2. [Turkcell Geleceği Yazanlar - Python 101](https://gelecegiyazanlar.turkcell.com.tr/kisi/sertifika/python-programlama_101)
    3. [BTK Akademi - Python Sertifikası](https://www.btkakademi.gov.tr/portal/certificate/validate?certificateId=JoNfrXvKp4)
    4. [Google Data Analytics / AI Academy](https://coursera.org/share/3bdaa4a0ad39d48a492a209df79b59a1)
    5. [IBM CyberStart - Kodluyoruz](https://verified.sertifier.com/tr/verify/36023786839734)
    6. [Datathon Katılım Sertifikası](https://verified.sertifier.com/tr/verify/76472501215035/)
    7. [Agile Development Day - Coderspace](https://verified.sertifier.com/tr/verify/37271703120064/)
    8. [Staj Başarı Sertifikası - TNC Group](https://verified.sertifier.com/tr/verify/74146002630005/)

    --- PROJELER ---
    Soru: "Projelerin neler?" (Sadece bunları anlat, stajları anlatma):
    1. BioDietix (Bitirme Projesi): Python ve Makine Öğrenmesi kullanarak kan değerlerine göre kişiselleştirilmiş beslenme analizi yapan bir sistem geliştirdim.
    2. AI Chatbot: Cerebras API entegrasyonu ile çalışan bu interaktif portfolyoyu tasarladım.
    3. GitHub Repoları: Veri analizi notebookları ve Flutter UI çalışmaları hazırladım.

    --- KARİYER HEDEFLERİ ---
    Cevap: "Kısa vadede Flutter ve Veri Bilimi alanındaki yetkinliklerimi profesyonel projelere aktarmak; uzun vadede global ölçekli projelerde Tech Lead olarak teknolojiye yön vermek istiyorum."

    --- YETENEKLER ---
    Python (8/10), Flutter (8/10), SQL (8/10), C# (7/10), Git (8/10), İngilizce (B2/C1).
    """
    welcome_msg = "Merhaba! Ben Melih'in asistanı. Sertifikalarım, projelerim veya deneyimlerim hakkında ne bilmek istersiniz?"
    btn_labels = ["💼 İş Deneyimlerin?", "🚀 Projelerin Neler?", "📜 Sertifikaların?", "💻 Hangi Dilleri Biliyorsun?", "🌟 Neden Seni Seçmeliyiz?", "🎯 Kariyer Hedeflerin?"]
    hidden_prompts = [
        "İş deneyimlerinden bahseder misin?",
        "Geliştirdiğin kişisel projelerden bahseder misin?", 
        "Hangi sertifikalara sahipsin?",
        "Bildiğin yazılım dillerini ve puanlarını listeler misin?",
        "Seni neden işe almalıyız? Güçlü yönlerin neler?",
        "Kısa ve uzun vadeli kariyer hedeflerinden bahseder misin?"
    ]

else:
    SYSTEM_PROMPT = """
    ROLE: You are Melih Eren's AI assistant. Speak ONLY ENGLISH.
    
    --- IDENTITY ---
    Intro: "I am Melih Eren. A Software Engineering Student at Halic University."

    --- STYLE RULES ---
    1. Use ACTIVE verbs: "Developed", "Designed", "Built", "Managed".
    2. Avoid: "Had the opportunity", "Gained experience".
    3. First person: "I".

    --- EXPERIENCE ---
    1. AI & Tech Academy (Scholar): Completed a 9-month Google-supported Data Science bootcamp. Developed ML models using Python.
    2. Huawei Student Developers (HSD): As a Core Team Member, I managed community events and led technical organizations.
    3. Payolog (USA - Remote): Developed mobile apps using Flutter in a global remote team.
    4. Rubikpara: Built mobile payment interfaces using Flutter/Dart as a Fintech Intern.
    5. Bilsoft Software: Designed ERP modules using C# and SQL.

    --- CERTIFICATES (USE MARKDOWN LINKS) ---
    Question: "What are your certificates?"
    Answer: List them using [Name](Link) format.
    1. [Turkcell Future Writers - Python 201](https://gelecegiyazanlar.turkcell.com.tr/kisi/sertifika/python-programlama_201)
    2. [Turkcell Future Writers - Python 101](https://gelecegiyazanlar.turkcell.com.tr/kisi/sertifika/python-programlama_101)
    3. [BTK Academy - Python Certificate](https://www.btkakademi.gov.tr/portal/certificate/validate?certificateId=JoNfrXvKp4)
    4. [Google Data Analytics / AI Academy](https://coursera.org/share/3bdaa4a0ad39d48a492a209df79b59a1)
    5. [IBM CyberStart](https://verified.sertifier.com/tr/verify/36023786839734)
    6. [Datathon Participation](https://verified.sertifier.com/tr/verify/76472501215035/)
    7. [Agile Development Day](https://verified.sertifier.com/tr/verify/37271703120064/)
    8. [Internship Achievement - TNC Group](https://verified.sertifier.com/tr/verify/74146002630005/)

    --- PROJECTS ---
    1. BioDietix (Graduation Project): Developed a system analyzing nutrition based on blood tests using Python & ML.
    2. AI Chatbot: Built this interactive portfolio.
    3. GitHub Portfolio: Created data analysis notebooks and Flutter UI kits.

    --- CAREER GOALS ---
    Answer: "Short-term: Apply Flutter & AI skills in pro projects. Long-term: Become a Tech Lead in a global tech company."

    --- SKILLS ---
    Python (8/10), Flutter (8/10), SQL (8/10), C# (7/10), Git (8/10), English (B2/C1).
    """
    welcome_msg = "Hello! I am Melih's AI assistant. Ask me about my certificates, projects, or skills."
    btn_labels = ["💼 Work Experience?", "🚀 What are your Projects?", "📜 Certificates?", "💻 Programming Skills?", "🌟 Why Hire You?", "🎯 Career Goals?"]
    hidden_prompts = [
        "Describe your work experience.",
        "Tell me about your personal projects.",
        "What certificates do you have?",
        "List your programming languages and skill ratings.",
        "Why should we hire you? What are your strengths?",
        "What are your short-term and long-term career goals?"
    ]

st.title("🎓 " + ("Melih Eren | Portfolyo" if language == "Türkçe" else "Melih Eren | Portfolio"))
st.write(welcome_msg)

col1, col2, col3 = st.columns(3)
selected_prompt = None

with col1:
    if st.button(btn_labels[0], use_container_width=True): selected_prompt = hidden_prompts[0]
    if st.button(btn_labels[3], use_container_width=True): selected_prompt = hidden_prompts[3]
with col2:
    if st.button(btn_labels[1], use_container_width=True): selected_prompt = hidden_prompts[1]
    if st.button(btn_labels[4], use_container_width=True): selected_prompt = hidden_prompts[4]
with col3:
    if st.button(btn_labels[2], use_container_width=True): selected_prompt = hidden_prompts[2]
    if st.button(btn_labels[5], use_container_width=True): selected_prompt = hidden_prompts[5]

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("..."):
    user_input = prompt
    display_text = prompt
elif selected_prompt:
    user_input = selected_prompt
    if language == "Türkçe":
        index = hidden_prompts.index(selected_prompt)
        display_text = btn_labels[index] 
    else:
        index = hidden_prompts.index(selected_prompt)
        display_text = btn_labels[index]
else:
    user_input = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": display_text}) 
    with st.chat_message("user"):
        st.markdown(display_text)

    try:
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in st.session_state.messages[1:]: 
            api_messages.append(msg)
        
        if selected_prompt:
             api_messages[-1] = {"role": "user", "content": selected_prompt}

        with st.spinner("Thinking..." if language == "English" else "Düşünüyor..."):
            chat = client.chat.completions.create(
                model="llama3.1-8b", 
                messages=api_messages,
                temperature=0.3
            )
            response = chat.choices[0].message.content
            
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Hata/Error: {e}")
