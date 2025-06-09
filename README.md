# 📄 GenAI PDF Q&A App

> Upload a PDF and get instant, accurate answers to your questions using GenAI-powered document intelligence.

---

## ✨ Overview

**GenAI PDF Q&A App** is an interactive web application built with Streamlit and powered by the Retrieval-Augmented Generation (RAG) architecture. It allows users to upload any PDF document, ask natural language questions, and receive accurate, contextual answers derived directly from the content of the PDF.

---

## 🚀 Features

- 📁 Upload any PDF document (up to 200MB)
- 🔍 Ask text-based questions and get intelligent, context-aware answers
- 🧠 Combines vector search with Ollama LLMs for powerful RAG-based Q&A
- 🌗 Toggle between Light and Dark themes
- 🙋‍♂️ Personalized footer with developer contact and credits

---

## 🛠 Tech Stack

| Component      | Technology              |
|----------------|-------------------------|
| Frontend       | Streamlit               |
| LLM            | Ollama (`gemma3`)       |
| Embeddings     | `nomic-embed-text`      |
| Vector Search  | FAISS                   |
| Parsing PDF    | LangChain PyPDFLoader   |
| Text Splitter  | RecursiveCharacterTextSplitter |
| Pipeline       | LangChain Prompt Chaining |

---

## 🧪 How It Works

1. **PDF Upload** – Drag and drop your file in the sidebar
2. **Text Splitting** – PDF is split into manageable chunks
3. **Embedding & Indexing** – Chunks are embedded and stored in FAISS
4. **Question Input** – You type your question
5. **RAG Chain** – Relevant chunks are retrieved, and LLM generates the answer
6. **Output** – The result is displayed in a clean, readable format

---

## 💡 Use Cases

- 📚 Study assistant for technical papers and books
- 🧾 Contract/document review automation
- 🧑‍💻 Personal knowledgebase search
- 💼 Business report summarization

---

## 📸 Demo Screenshot

![Screenshot 2025-06-10 015340](https://github.com/user-attachments/assets/53eec4c8-0e18-426e-bfe2-3a1bf7713993)

---

## 👨‍💻 Developer

Made with ❤️ by **Pranav Vashisth**

- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/pranav-vashisth/)
- 📬 Email: [pvashisth0711@gmail.com](mailto:pvashisth0711@gmail.com)

---

## 🏁 Getting Started

1. Clone this repository  
   ```bash
   git clone https://github.com/pvashisth-07/GenAI-Pdf-QnA-Bot.git
   cd GenAI-Pdf-QnA-Bot
