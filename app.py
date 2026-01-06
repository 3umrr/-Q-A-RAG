"""
Q&A RAG Application with PDF Support using Groq LLM and Streamlit

This application allows users to:
1. Upload PDF documents and create a vector store
2. Ask questions about the documents
3. Get AI-powered answers using Groq's Gemma model

Requirements:
- Groq API Key (GROQ_API_KEY) - Free from https://console.groq.com
- Hugging Face Embeddings (Free, no API key needed, runs locally)

Environment Setup:
Create a .env file with:
    GROQ_API_KEY=your_groq_api_key
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from io import BytesIO
import tempfile


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Configure Streamlit page
st.set_page_config(
    page_title="📄 Q&A with PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_api_keys():
    """Validate that all required API keys are set."""
    if not groq_api_key:
        st.error(
            "❌ **GROQ_API_KEY not found!**\n\n"
            "Please add your Groq API key to the `.env` file:\n"
            "`GROQ_API_KEY=your_key_here`"
        )
        st.stop()


# Run validation
validate_api_keys()


# ============================================================================
# INITIALIZE LLM & EMBEDDINGS
# ============================================================================

@st.cache_resource
def load_llm():
    """Initialize and cache the Groq LLM model."""
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.7
    )


# Load the LLM
llm = load_llm()

# Display page title
st.title("📄 Q&A with your PDFs")
st.markdown("Ask questions about your PDF documents and get AI-powered answers")


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

qa_prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    Context: {context}
    
    Question: {input}
    
    Answer:"""
)


# ============================================================================
# VECTOR STORE FUNCTIONS
# ============================================================================

def vector_embedding(pdf_files):
    """
    Create vector store from uploaded PDF files.
    
    Args:
        pdf_files: List of uploaded PDF file objects from Streamlit
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not pdf_files:
        st.error("❌ No PDF files provided")
        return False
    
    try:
        with st.spinner("📥 Loading and processing PDFs..."):
            # Initialize embeddings using Hugging Face (free, no API key needed)
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Load PDFs from uploaded files
            st.session_state.docs = []
            
            for pdf_file in pdf_files:
                # Create temporary file from uploaded bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Load PDF using PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    st.session_state.docs.extend(docs)
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)
            
            # Validate documents were loaded
            if not st.session_state.docs:
                st.error("❌ No content found in the uploaded PDFs")
                return False
            
            st.success(f"✅ Loaded {len(st.session_state.docs)} pages from PDF(s)")
            
            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(
                    st.session_state.docs
                )
            )
            
            # Validate chunks were created
            if not st.session_state.final_documents:
                st.warning("⚠️ PDF content is too small to split into chunks. Using original documents...")
                st.session_state.final_documents = st.session_state.docs
            
            st.success(
                f"✅ Split into {len(st.session_state.final_documents)} chunks"
            )
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            
            st.session_state.pdf_uploaded = True
            return True
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return False


# ============================================================================
# SIDEBAR - PDF UPLOAD
# ============================================================================

with st.sidebar:
    st.header("📁 Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Select PDF files to analyze",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to create a searchable document database"
    )
    
    if uploaded_files:
        if st.button("🚀 Process PDFs", key="process_pdfs"):
            # Clear existing vectors to force reload
            if "vectors" in st.session_state:
                del st.session_state.vectors
            
            success = vector_embedding(uploaded_files)
            if success:
                st.success("✅ PDFs processed and ready for Q&A!")
    else:
        st.info("👆 Upload PDF files to get started")


# ============================================================================
# MAIN CONTENT - Q&A INTERFACE
# ============================================================================

st.header("💬 Ask Questions About Your Documents")

# Check if vector store exists
if "vectors" not in st.session_state:
    st.info(
        "📌 **Get Started:**\n\n"
        "1. Upload your PDF files using the uploader in the sidebar\n"
        "2. Click 'Process PDFs'\n"
        "3. Then ask your questions here!"
    )
else:
    st.success("✅ Vector store loaded and ready!")

# User input
st.subheader("Enter Your Question")
user_question = st.text_input(
    label="Your question:",
    placeholder="What information is in the documents?",
    label_visibility="collapsed"
)

# Process question
if user_question:
    if "vectors" not in st.session_state:
        st.warning(
            "⚠️ Please upload and process PDFs first using the "
            "file uploader in the sidebar."
        )
    else:
        try:
            with st.spinner("🤔 Searching documents and generating answer..."):
                # Create retriever
                retriever = st.session_state.vectors.as_retriever()
                
                # Format retrieved documents
                def format_docs(docs):
                    """Format retrieved documents for the prompt."""
                    return "\n\n".join(
                        f"Document {i+1}:\n{doc.page_content}"
                        for i, doc in enumerate(docs)
                    )
                
                # Create RAG chain
                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "input": RunnablePassthrough()
                    }
                    | qa_prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Measure response time
                start_time = time.time()
                
                # Generate response
                response = rag_chain.invoke(user_question)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
            
            # Display answer
            st.subheader("✅ Answer")
            st.markdown(response)
            
            # Display metadata
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
            with col2:
                st.caption(f"📝 Response length: {len(response)} characters")
        
        except Exception as error:
            st.error(f"❌ Error processing question: {str(error)}")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    📄 Q&A RAG Application | Made with ❤️ using Streamlit & Groq
    </div>
    """,
    unsafe_allow_html=True
)
