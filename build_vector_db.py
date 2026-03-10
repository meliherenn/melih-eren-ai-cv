from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def create_vector_db():
    """
    Loads all available CV PDFs (Turkish & English),
    splits them into chunks, and builds a unified FAISS vector index.
    """
    pdf_files = [
        "Melih_Eren_cvtr.pdf",       # Turkish CV
        "Melih_Eren_ATS_CV.pdf",     # English ATS CV
    ]

    all_documents = []

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"📄 Okunuyor: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Add language metadata to each document
            lang = "tr" if "cvtr" in pdf_path.lower() else "en"
            for doc in docs:
                doc.metadata["language"] = lang
                doc.metadata["source_file"] = pdf_path
            all_documents.extend(docs)
            print(f"   ✅ {len(docs)} sayfa okundu.")
        else:
            print(f"   ⚠️ Atlanıyor (bulunamadı): {pdf_path}")

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
    vectorstore.save_local("faiss_index")
    print("✅ İşlem tamam! 'faiss_index' klasörü oluşturuldu/güncellendi.")

if __name__ == "__main__":
    create_vector_db()
