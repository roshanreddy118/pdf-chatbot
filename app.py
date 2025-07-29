
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

# ------------------------ Setup Functions ------------------------

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(raw_docs)

def build_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding_model)

def load_llm():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

def setup_qa_chain(pdf_path):
    docs = load_pdf(pdf_path)
    db = build_vectorstore(docs)
    llm = load_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# ------------------------ Initialize Once ------------------------

qa_chain = setup_qa_chain("sample_doc.pdf")  # 🔁 Replace with your PDF path

# ------------------------ Chatbot Logic ------------------------

def chatbot(question, chat_history):
    result = qa_chain.invoke({"query": question})
    return result["result"]  # ✅ Gradio expects a plain string

# ------------------------ Launch Gradio ------------------------

chat_ui = gr.ChatInterface(
    fn=chatbot,
    title="📄 PDF ChatBot (Flan-T5)",
    theme="default"
)
chat_ui.launch()
