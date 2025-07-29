# 🤖 PDF Chatbot using Hugging Face + LangChain

This is a simple PDF-based QA chatbot built with:
- 🧠 Hugging Face models (Zephyr)
- 🔍 LangChain for RAG pipeline
- 🗂️ FAISS for vector storage
- 💬 Gradio for chat UI

| Model            | Speed  | Size    | GPU Needed     |
| ---------------- | ------ | ------- | -------------- |
| `flan-t5-base`   | Fast   | \~250MB | ❌ Not required |
| `flan-t5-large`  | Medium | \~800MB | ⚠️ Maybe       |
| `zephyr-7b-beta` | Slow   | \~13GB  | ✅ Required     |

## 📦 Setup

```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
