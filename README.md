# 🚀 Melih Eren | AI Portfolio Chatbot

An interactive AI-powered portfolio chatbot built with **Streamlit**, **Cerebras AI (LLaMA 3.3)**, and **RAG (FAISS)** for intelligent, context-aware conversations about my CV.

🌐 **Live Demo:** [melih-eren-ai-cv.streamlit.app](https://melih-eren-ai-cv.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Chatbot** | Powered by Cerebras AI (LLaMA 3.3-70B) for natural conversations |
| 📄 **RAG System** | FAISS vector database for PDF-based context retrieval |
| 🌍 **Bilingual** | Full Turkish & English support with language-aware responses |
| 📥 **CV Download** | Language-specific PDF download (TR/EN) |
| 🔐 **Admin Panel** | Secure panel to edit all CV data (profile, experience, projects, skills, certificates) |
| 🎨 **Premium UI** | Glassmorphism design with animated gradients and micro-interactions |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **AI/LLM:** Cerebras AI API (LLaMA 3.3-70B)
- **RAG:** LangChain + FAISS + sentence-transformers
- **Styling:** Custom CSS (Glassmorphism, Gradient Animations)
- **Language:** Python

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/meliherenn/melih-eren-ai-cv.git
cd melih-eren-ai-cv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
CEREBRAS_API_KEY = "your-cerebras-api-key"
ADMIN_PASSWORD = "your-admin-password"
EOF

# 4. Build the vector database
python build_vector_db.py

# 5. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
cv-bot/
├── app.py                  # Main Streamlit application
├── build_vector_db.py      # Vector DB builder (processes TR & EN PDFs)
├── data.json               # CV data (bilingual)
├── style.css               # Custom CSS styles
├── requirements.txt        # Python dependencies
├── ben.png                 # Profile photo
├── Melih_Eren_cvtr.pdf     # Turkish CV
├── Melih_Eren_ATS_CV.pdf   # English ATS CV
├── faiss_index/            # FAISS vector database
│   ├── index.faiss
│   └── index.pkl
└── .streamlit/
    ├── config.toml         # Streamlit theme config
    └── secrets.toml        # API keys (not committed)
```

---

## 🔑 Environment Variables

| Key | Description |
|-----|-------------|
| `CEREBRAS_API_KEY` | API key from [Cerebras AI](https://cloud.cerebras.ai/) |
| `ADMIN_PASSWORD` | Password for the admin panel |

---

## 👨‍💻 Author

**Melih Eren** — Software Engineering Student @ Halic University

- 🐙 [GitHub](https://github.com/meliherenn)
- 💼 [LinkedIn](https://www.linkedin.com/in/meliheren/)
- 📧 meliheren2834@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
