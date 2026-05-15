import os
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-top: 0.5rem;
    font-size: 1.2rem;
}

.hero-section {
    padding-top: 5rem;
    padding-bottom: 3rem;
}

.stFileUploader {
    border: 2px dashed #3b82f6;
    padding: 1rem;
    border-radius: 15px;
    background-color: rgba(255,255,255,0.03);
}

.stTextInput > div > div > input {
    border-radius: 12px;
    background-color: #111827;
    color: white;
    padding: 12px;
}

.answer-box {
    background-color: rgba(255,255,255,0.05);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #3b82f6;
    margin-top: 1rem;
}

.history-card {
    background-color: rgba(255,255,255,0.04);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Hero Section
st.markdown(
    """
    <div class="hero-section">
        <h1 class="main-title">🤖 DocuMind AI</h1>
        <p class="subtitle">
            Chat with your PDFs using AI-powered document intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload Section
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type="pdf"
)

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📚 Reading and understanding your PDF..."):

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector Store
        vectorstore = Chroma.from_documents(
            docs,
            embeddings
        )

        retriever = vectorstore.as_retriever()

        # LLM
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

    st.success("✅ PDF Ready! Ask your questions below.")

    # Question Input
    question = st.text_input(
        "💬 Ask anything about the document"
    )

    if question:

        with st.spinner("🤖 Generating answer..."):

            # Retrieve relevant docs
            relevant_docs = retriever.invoke(question)

            context = "\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            # Prompt
            prompt = f"""
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {question}
            """

            # Generate response
            response = llm.invoke(prompt)

        # Save history
        st.session_state.history.append({
            "question": question,
            "answer": response.content
        })

        # Response Box
        st.subheader("🤖 AI Response")

        st.markdown(
            f"""
            <div class="answer-box">
                {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")

st.markdown(
    """
    <center>
        Built using LangChain + Groq + Streamlit
    </center>
    """,
    unsafe_allow_html=True
)