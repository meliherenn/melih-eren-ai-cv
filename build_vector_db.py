
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def create_vector_db():
    pdf_path = "MELİH EREN.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Hata: {pdf_path} bulunamadı!")
        return

    print("PDF okunuyor...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"{len(documents)} sayfa okundu.")
    
    print("Metin bölünüyor...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"{len(texts)} parçaya bölündü.")
    
    print("Vektörler oluşturuluyor (Bu işlem biraz sürebilir)...")
    # Using a lightweight, free embedding model that runs locally CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    print("Vektör veritabanı kaydediliyor...")
    vectorstore.save_local("faiss_index")
    print("İşlem tamam! 'faiss_index' klasörü oluşturuldu.")

if __name__ == "__main__":
    create_vector_db()
