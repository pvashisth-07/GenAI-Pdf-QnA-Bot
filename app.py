'''
GenAI PDF Q&A is an intelligent question-answering web application that allows users to upload any PDF document and ask natural language questions about its content. Powered by RAG (Retrieval-Augmented Generation) architecture, the system leverages Ollama LLMs and FAISS vector search to deliver precise, context-aware answers directly sourced from the uploaded document.

Made by: Pranav Vashith
'''

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

st.set_page_config(page_title="PDF Q&A - GenAI by Pranav", layout="wide")

with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    st.markdown("---")
    theme = st.radio("Choose Theme", ["Light", "Dark"])

    if theme == "Dark":
        st.markdown("""
            <style>
                body, .stApp { background-color: #1e1e1e; color: white; }
                .stTextInput > div > div > input { color: black !important; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                body, .stApp { background-color: #f9f9f9; color: black; }
            </style>
        """, unsafe_allow_html=True)

st.title("📄 PDF Q&A App")
st.caption("Ask anything from your uploaded PDF!")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        data = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_data = splitter.split_documents(data)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.from_documents(final_data, embeddings)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions based on the context provided below.

<context>
{context}
</context>

Instructions:
- Use the context above as your primary source of truth.
- If the context provides a clear answer, answer strictly from it.
- If the context gives a slight mention or hint of a topic, you may use minimal external knowledge to clarify or expand the answer.
- If there is no relevant information in the context at all, say: "The answer is not found in the document."
"""),
            ("user", "Question: {question}")
        ])
        llm = OllamaLLM(model="gemma3")
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        st.success("PDF processed successfully! You can now ask questions.")

        question = st.text_input("Ask a question from the PDF:")
        if question:
            with st.spinner("Searching for answer..."):
                docs = db.similarity_search(question, k=3)
                context = "\n\n".join(doc.page_content for doc in docs)
                response = chain.invoke({"context": context, "question": question})
                st.markdown("### 🧠 Answer:")
                st.write(response)

st.markdown("""
<hr style="margin-top: 50px; border-top: 1px solid #bbb;">
<div style="text-align: center; font-size: 15px;">
    Made with ❤️ by <strong>Pranav Vashisth</strong><br>
    <a href="https://www.linkedin.com/in/pranav-vashisth/" target="_blank">LinkedIn</a> |
    <a href="mailto:pvashisth0711@gmail.com">Email</a>
</div>
""", unsafe_allow_html=True)
