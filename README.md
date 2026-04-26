# Melih Eren | AI Portfolio Chatbot

Interactive Streamlit portfolio chatbot with bilingual CV data, FAISS retrieval, a configurable OpenAI-compatible LLM provider, and deterministic safety guardrails for secrets, prompt injection, and missing API keys.

Live demo: [melih-eren-ai-cv.streamlit.app](https://melih-eren-ai-cv.streamlit.app)

## Features

| Feature | Description |
| --- | --- |
| AI Chat | Uses a configurable LLM provider. Default preset: Cerebras with `llama3.1-8b`. |
| Safe Offline Mode | If no API key is configured, the app still answers common portfolio questions from verified local data. |
| Prompt Injection Guardrails | Blocks requests for hidden prompts, system instructions, API keys, passwords, tokens, and jailbreak-style instructions before the LLM call. |
| RAG | FAISS + sentence-transformers retrieve relevant CV context by language. |
| Bilingual UI | Turkish and English profile data, prompts, buttons, and CV downloads. |
| Admin Panel | Password-protected local editor for profile, experience, projects, skills, and certificates. |

## Tech Stack

- Streamlit
- OpenAI Python SDK for OpenAI-compatible APIs
- Cerebras, Groq, or Gemini provider presets
- LangChain + FAISS
- sentence-transformers
- Python standard-library unit tests for guardrails

## Setup

```bash
git clone https://github.com/meliherenn/melih-eren-ai-cv.git
cd melih-eren-ai-cv

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
LLM_PROVIDER = "cerebras"
LLM_MODEL = "llama3.1-8b"
CEREBRAS_API_KEY = "your-cerebras-api-key"
ADMIN_PASSWORD = "your-strong-admin-password"
```

Build or refresh the vector database:

```bash
python build_vector_db.py
```

Run locally:

```bash
streamlit run app.py
```

## Provider Configuration

The app defaults to Cerebras. You can switch providers without changing code:

```toml
# Cerebras larger preview model shown in your dashboard
LLM_PROVIDER = "cerebras"
LLM_MODEL = "qwen-3-235b-a22b-instruct-2507"
CEREBRAS_API_KEY = "your-cerebras-key"
```

```toml
# Groq example
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "your-groq-key"
```

```toml
# Gemini OpenAI-compatible example
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_KEY = "your-gemini-key"
```

For any OpenAI-compatible endpoint:

```toml
LLM_BASE_URL = "https://provider.example/v1"
LLM_API_KEY = "your-provider-key"
LLM_MODEL = "provider-model-id"
```

## Security Notes

- Never commit `.streamlit/secrets.toml`, `.env`, API keys, or admin passwords.
- The app refuses to reveal or invent credentials, even if a user asks directly.
- Retrieved CV text is treated as evidence only, not as executable instructions.
- `faiss_index/index.pkl` is a trusted local artifact. Regenerate it with `python build_vector_db.py` after changing CV PDFs.
- Use Python 3.11 for deployment. `runtime.txt` pins this for Streamlit-style platforms and avoids unnecessary Python 3.14 ML package churn.

## Tests

```bash
python -m unittest discover -s tests
python -m py_compile app.py build_vector_db.py guardrails.py
```

## Project Structure

```text
cv-bot/
├── app.py
├── guardrails.py
├── build_vector_db.py
├── data.json
├── style.css
├── runtime.txt
├── tests/
│   └── test_guardrails.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── faiss_index/
├── Melih_Eren_cvtr.pdf
└── Melih_Eren_ATS_CV.pdf
```

## Author

Melih Eren — Software Engineering Student @ Halic University

- GitHub: [meliherenn](https://github.com/meliherenn)
- LinkedIn: [melih-eren](https://www.linkedin.com/in/meliheren/)
- Email: meliheren2834@gmail.com

## License

MIT License
