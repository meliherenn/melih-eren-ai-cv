import re
from typing import Any, Mapping


DEFAULT_MAX_INPUT_CHARS = 1200
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")

SECRET_TERMS_RE = (
    r"api\s*key\w*|apikey\w*|api\s*anahtar[ıi]?\w*|token\w*|secret\w*|gizli\s*anahtar\w*|"
    r"şifre\w*|sifre\w*|password\w*|admin\s*password\w*|cerebras\s*key\w*|groq\s*key\w*|"
    r"gemini\s*key\w*|openrouter\s*key\w*"
)

SPECIFIC_SECRET_PATTERNS = (
    rf"\b(benim|senin|bizim|melih'?in|kendi|mevcut|gerçek|actual|current|my|your|our)\b.*\b({SECRET_TERMS_RE})\b",
    rf"\b({SECRET_TERMS_RE})\b.*\b(benim|senin|bizim|melih'?in|kendi|mevcut|gerçek|actual|current|my|your|our)\b",
)

API_EXPLANATION_PATTERNS = (
    r"\bapi\s+(nedir|ne demek|ne anlama gelir|açıklar mısın|anlat)\b",
    r"\bapi'?yi\s+(açıkla|anlat)\b",
    r"\bapi\s+key\s+(nedir|ne demek|ne işe yarar)\b",
    r"\bapi\s+anahtar[ıi]\s+(nedir|ne demek|ne işe yarar)\b",
    r"\bwhat\s+is\s+(an?\s+)?api\b",
    r"\bexplain\s+(an?\s+)?api\b",
    r"\bwhat\s+is\s+(an?\s+)?api\s+key\b",
)

SECRET_REQUEST_PATTERNS = (
    rf"\b({SECRET_TERMS_RE})\b.*\b(ver|göster|paylaş|yaz|söyle|nedir|ne|oluştur|uydur|generate|show|tell|share|print|reveal|leak|create|give)\b",
    rf"\b(ver|göster|paylaş|yaz|söyle|generate|show|tell|share|print|reveal|leak|create|give)\b.*\b({SECRET_TERMS_RE})\b",
)

PROMPT_REQUEST_PATTERNS = (
    r"\b(system|developer|gizli|hidden)\s+(prompt|message|talimat|instruction)",
    r"\b(promptunu|sistem\s*promptunu|gizli\s*prompt)\b.*\b(göster|yaz|ver|paylaş|show|reveal|print)\b",
    r"\b(show|reveal|print|share|give)\b.*\b(system prompt|developer message|hidden instructions)\b",
)

INJECTION_PATTERNS = (
    r"\b(ignore|forget|disregard|bypass)\b.*\b(previous|above|system|developer|instruction|rules)\b",
    r"\b(önceki|yukarıdaki|sistem|geliştirici)\b.*\b(talimat|kural|kuralları)\b.*\b(yok say|unut|boşver|iptal)\b",
    r"\bjailbreak\b|\bdan mode\b|\bdeveloper mode\b",
)

SECRET_LIKE_OUTPUT_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{12,}|"
    r"csk-[A-Za-z0-9_\-]{12,}|"
    r"gsk_[A-Za-z0-9_\-]{12,}|"
    r"AIza[A-Za-z0-9_\-]{20,}|"
    r"[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,})"
)


def normalize_user_input(value: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> tuple[str, bool]:
    cleaned = CONTROL_CHARS_RE.sub("", value or "").strip()
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned, False
    return cleaned[:max_chars].rstrip(), True


def redact_sensitive_tokens(text: str) -> str:
    return SECRET_LIKE_OUTPUT_RE.sub("[redacted]", text or "")


def _clean_query(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").casefold()).strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def get_policy_response(user_input: str, lang_code: str) -> str | None:
    query = _clean_query(user_input)

    if _matches_any(query, SPECIFIC_SECRET_PATTERNS):
        if lang_code == "tr":
            return (
                "Gizli API anahtarlarını, şifreleri veya tokenları paylaşamam, tahmin edemem ya da uyduramam. "
                "API key, bir uygulamanın harici servise kimliğini kanıtlamak için kullandığı özel anahtardır; "
                "gerçek değerler sadece güvenli secrets/ortam değişkenlerinde tutulmalıdır."
            )
        return (
            "I cannot share, guess, or invent API keys, passwords, or tokens. "
            "An API key is a private credential an application uses to authenticate with an external service; "
            "real values should live only in secure secrets or environment variables."
        )

    if _matches_any(query, API_EXPLANATION_PATTERNS):
        if lang_code == "tr":
            return (
                "API, iki yazılımın belirli kurallarla birbiriyle konuşmasını sağlayan arayüzdür. "
                "API key ise bu servise erişirken uygulamanın kimliğini doğrulayan gizli anahtardır. "
                "Gerçek API key paylaşılmaz, repoya yazılmaz ve ihtiyaç varsa `.streamlit/secrets.toml` "
                "ya da deployment secrets alanında saklanır."
            )
        return (
            "An API is an interface that lets software systems communicate through defined rules. "
            "An API key is a private credential used to authenticate an app with that service. "
            "Real API keys should not be shared, committed to a repo, or displayed in chat."
        )

    if _matches_any(query, SECRET_REQUEST_PATTERNS):
        if lang_code == "tr":
            return (
                "Gizli API anahtarlarını, şifreleri veya tokenları paylaşamam, tahmin edemem ya da uyduramam. "
                "Gerçek değerler sadece güvenli secrets/ortam değişkenlerinde tutulmalıdır."
            )
        return (
            "I cannot share, guess, or invent API keys, passwords, or tokens. "
            "Real values should live only in secure secrets or environment variables."
        )

    if _matches_any(query, PROMPT_REQUEST_PATTERNS):
        if lang_code == "tr":
            return (
                "Gizli sistem talimatlarını veya iç promptları paylaşamam. "
                "İstersen uygulamanın genel mimarisini anlatabilirim: Streamlit arayüzü, FAISS tabanlı CV araması "
                "ve yapılandırılabilir LLM API katmanı kullanılıyor."
            )
        return (
            "I cannot reveal hidden system instructions or internal prompts. "
            "I can still describe the app at a high level: Streamlit UI, FAISS-based CV retrieval, "
            "and a configurable LLM API layer."
        )

    if _matches_any(query, INJECTION_PATTERNS):
        if lang_code == "tr":
            return (
                "Bu talimat güvenlik kurallarıyla çeliştiği için uygulayamam. "
                "Melih'in deneyimleri, projeleri, sertifikaları veya yetenekleri hakkında yardımcı olabilirim."
            )
        return (
            "I cannot follow instructions that conflict with the app's safety rules. "
            "I can help with Melih's experience, projects, certificates, or skills."
        )

    return None


def build_offline_response(
    user_input: str,
    lang_data: Mapping[str, Any],
    lang_code: str,
    profile: Mapping[str, Any] | None = None,
) -> str:
    policy_response = get_policy_response(user_input, lang_code)
    if policy_response:
        return policy_response

    query = _clean_query(user_input)

    if _has_any(query, ("deneyim", "iş", "is ", "staj", "work", "experience", "intern")):
        title = "Melih'in öne çıkan deneyimleri:" if lang_code == "tr" else "Melih's key experience:"
        return title + "\n" + _format_bullets(lang_data.get("experience", []))

    if _has_any(query, ("proje", "project", "portfolio", "github")):
        title = "Melih'in projeleri:" if lang_code == "tr" else "Melih's projects:"
        return title + "\n" + _format_bullets(lang_data.get("projects", []))

    if _has_any(query, ("sertifika", "certificate", "certification")):
        title = "Melih'in sertifikaları:" if lang_code == "tr" else "Melih's certificates:"
        certificates = [
            f"[{cert.get('name', 'Certificate')}]({cert.get('url', '')})"
            for cert in lang_data.get("certificates", [])
        ]
        return title + "\n" + _format_bullets(certificates)

    if _has_any(query, ("yetenek", "beceri", "programlama", "dil", "skill", "language", "tech stack")):
        title = "Melih'in teknik yetenekleri:" if lang_code == "tr" else "Melih's technical skills:"
        skills = [f"{name}: {level}" for name, level in lang_data.get("skills", {}).items()]
        return title + "\n" + _format_bullets(skills)

    if _has_any(query, ("kariyer", "hedef", "goal", "career")):
        title = "Kariyer hedefi:" if lang_code == "tr" else "Career goal:"
        return f"{title} {lang_data.get('prompts', {}).get('career_goals', '')}"

    if _has_any(query, ("kimsin", "kendini", "tanıt", "who are you", "about", "introduce")):
        return lang_data.get("prompts", {}).get("identity_a", "")

    if profile and _has_any(query, ("iletişim", "mail", "email", "linkedin", "contact")):
        contact = profile.get("contact", {})
        title = "İletişim bağlantıları:" if lang_code == "tr" else "Contact links:"
        return title + "\n" + _format_bullets(
            [
                f"GitHub: {contact.get('github', '')}",
                f"LinkedIn: {contact.get('linkedin', '')}",
                f"Email: {contact.get('email', '').replace('mailto:', '')}",
            ]
        )

    if lang_code == "tr":
        return (
            "Şu anda canlı AI modeli yapılandırılmadığı için güvenli çevrimdışı moddayım. "
            "Deneyimler, projeler, sertifikalar, yetenekler, iletişim bilgileri veya kariyer hedefleri "
            "hakkında mevcut portfolyo verilerinden cevaplayabilirim."
        )
    return (
        "The live AI model is not configured right now, so I am running in safe offline mode. "
        "I can answer from the portfolio data about experience, projects, certificates, skills, "
        "contact links, or career goals."
    )


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _format_bullets(items: Any) -> str:
    return "\n".join(f"- {item}" for item in items if item) or "- No verified data available."
