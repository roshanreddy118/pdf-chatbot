import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ----------------- Core Functions ----------------- #

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def load_llm():
    gen_pipeline = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=gen_pipeline)

def create_qa_chain(vectorstore, llm):
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# ----------------- Setup ----------------- #

pdf_path = "sample.pdf"  # Replace with your PDF
documents = load_pdf(pdf_path)
vectorstore = create_vector_store(documents)
llm = load_llm()
qa_chain = create_qa_chain(vectorstore, llm)

# ----------------- Chat Function ----------------- #

def chatbot(user_input, history=[]):
    answer = qa_chain.run(user_input)
    history.append((user_input, answer))
    return history, history

# ----------------- Gradio UI ----------------- #

chat_ui = gr.ChatInterface(fn=chatbot, chatbot=True, title="PDF Chatbot - Powered by Hugging Face")
if __name__ == "__main__":
    chat_ui.launch()
