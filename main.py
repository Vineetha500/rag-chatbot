import os
import streamlit as st
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama.chat_models import ChatOllama
vector_space_dir = os.path.join(os.getcwd(), "vector_db")
if not os.path.exists(vector_space_dir):
    os.mkdir(vector_space_dir)
st.set_page_config(page_title="RAG ChatBot", layout="centered")
st.title("RAG ChatBot (LangChain + LLaMA2)")
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
upload_pdf = st.file_uploader(
    "Upload the PDF file",
    type=["pdf"],
    key='upload_pdf'
)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
if upload_pdf is not None and st.session_state['vectorstore'] is None:
    with st.spinner("Loading PDF and creating vector DB...."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        vectorstore = FAISS.from_documents(
            documents,
            embedding_model
        )
        vectorstore.save_local(vector_space_dir)
        st.session_state['vectorstore'] = vectorstore
        st.session_state['retriever'] = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )
        st.success("Vector DB Created")
llm = ChatOllama(model="llama2")
if st.session_state['retriever'] is not None:

    custom_prompt = """
You are a helpful AI assistant.

Use the provided PDF context to answer the user's question.

Instructions:
- Give detailed and well-explained answers.
- Explain concepts clearly.
- Include important points from the document.
- If possible, provide examples.
- If the answer is not available in the PDF, say:
  "The information is not available in the uploaded document."

Question:
{question}
"""
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False
    )
    user_question = st.text_input("Ask your question:", key='text')
    if user_question:
        detailed_question = custom_prompt.format(
            question=user_question
        )
        with st.spinner("Thinking...."):
            result = qa_chain.run({
                "question": detailed_question
            })
        st.markdown(
            f"""
            <div style="
                background-color:#1e1e1e;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
            ">
                <h4 style="color:#00ffff;">You:</h4>
                <p style="color:white;">{user_question}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="
                background-color:#262730;
                padding:20px;
                border-radius:10px;
                border-left:5px solid #00ff99;
            ">
                <h4 style="color:#00ff99;">Bot:</h4>
                <p style="
                    color:white;
                    font-size:16px;
                    line-height:1.8;
                    text-align:justify;
                ">
                    {result}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def del_uploaded_pdf(path):
    if os.path.exists(path) and path:
        os.remove(path)
if st.button("Clear Session"):

    st.session_state['memory'].clear()
    st.session_state['retriever'] = None
    st.session_state['vectorstore'] = None
    del_vectordb(vector_space_dir)
    pdf_p = st.session_state.get('pdf_file_path', None)
    del_uploaded_pdf(pdf_p)
    st.session_state['pdf_file_path'] = None
    for key in ['upload_pdf', 'text']:
        if key in st.session_state:
            del st.session_state[key]
    st.success('Session, PDF and VectorDB are cleared')
    st.rerun()

