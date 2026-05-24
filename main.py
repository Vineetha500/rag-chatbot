import os
import streamlit as st
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
vector_space_dir = os.path.join(os.getcwd(), "vector_db")
os.makedirs(vector_space_dir, exist_ok=True)
st.set_page_config(page_title="RAG ChatBot", layout="centered")
st.title("RAG ChatBot (TinyLlama)")
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
upload_pdf = st.file_uploader("Upload PDF", type=["pdf"])
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
if upload_pdf and st.session_state["vectorstore"] is None:
    with st.spinner("Processing PDF..."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        documents = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(vector_space_dir)
        st.session_state["vectorstore"] = vectorstore
        st.session_state["retriever"] = vectorstore.as_retriever(
            search_kwargs={"k": 2}
        )
        st.success("Vector DB Ready")
llm = ChatOllama(model="tinyllama")
if st.session_state["retriever"] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state["retriever"],
        memory=st.session_state["memory"],
        return_source_documents=False
    )
    user_question = st.text_input("Ask your question:", key="question_input")
    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({
                "question": user_question
            })
            answer = result["answer"]
        st.markdown("### You")
        st.write(user_question)
        st.markdown("### Bot")
        st.write(answer)
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
if st.button("Clear Session"):
    st.session_state["memory"].clear()
    st.session_state["vectorstore"] = None
    st.session_state["retriever"] = None
    clear_folder(vector_space_dir)
    st.success("Cleared")
    st.rerun()
