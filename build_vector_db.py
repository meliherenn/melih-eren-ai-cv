from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "data.json"


def get_configured_pdfs():
    """Read active CV PDF paths from data.json instead of hardcoding filenames."""
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    profile = data.get("profile", {})
    return [
        (profile.get("cv_pdf_tr"), "tr"),
        (profile.get("cv_pdf_en"), "en"),
    ]


def create_vector_db():
    """
    Loads configured CV PDFs (Turkish & English),
    splits them into chunks, and builds a unified FAISS vector index.
    """
    pdf_files = get_configured_pdfs()

    all_documents = []

    for pdf_name, lang in pdf_files:
        if not pdf_name:
            continue

        pdf_path = (APP_ROOT / pdf_name).resolve()
        try:
            pdf_path.relative_to(APP_ROOT)
        except ValueError:
            print(f"   ⚠️ Atlanıyor (proje dışı dosya yolu): {pdf_name}")
            continue

        if pdf_path.exists():
            print(f"📄 Okunuyor: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["language"] = lang
                doc.metadata["source_file"] = pdf_path.name
            all_documents.extend(docs)
            print(f"   ✅ {len(docs)} sayfa okundu.")
        else:
            print(f"   ⚠️ Atlanıyor (bulunamadı): {pdf_path.name}")

    if not all_documents:
        print("❌ Hiçbir PDF bulunamadı! Lütfen PDF dosyalarını proje klasörüne ekleyin.")
        return

    print(f"\n📊 Toplam {len(all_documents)} sayfa yüklendi.")

    print("✂️  Metin bölünüyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"   📦 {len(texts)} parçaya bölündü.")

    print("🧠 Vektörler oluşturuluyor (bu işlem biraz sürebilir)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(texts, embeddings)

    print("💾 Vektör veritabanı kaydediliyor...")
    vectorstore.save_local(str(APP_ROOT / "faiss_index"))
    print("✅ İşlem tamam! 'faiss_index' klasörü oluşturuldu/güncellendi.")


if __name__ == "__main__":
    create_vector_db()
