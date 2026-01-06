"""
Q&A RAG Application with PDF Support using Groq LLM and Streamlit

This application allows users to:
1. Upload PDF documents and create a vector store
2. Ask questions about the documents
3. Get AI-powered answers using Groq's Gemma model

Requirements:
- Groq API Key (GROQ_API_KEY)
- Google API Key (GOOGLE_API_KEY)

Environment Setup:
Create a .env file with:
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
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
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set Google API key in environment
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

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
    
    if not google_api_key:
        st.error(
            "❌ **GOOGLE_API_KEY not found!**\n\n"
            "Please add your Google API key to the `.env` file:\n"
            "`GOOGLE_API_KEY=your_key_here`"
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
        model_name="Gemma-7b-it",
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

def vector_embedding():
    """
    Create vector store from PDFs in ./us_census directory.
    Stores result in st.session_state for reuse across interactions.
    """
    if "vectors" not in st.session_state:
        try:
            with st.spinner("📥 Loading and processing PDFs..."):
                # Initialize embeddings
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
                
                # Load PDFs from directory
                st.session_state.loader = PyPDFDirectoryLoader("./us_census")
                st.session_state.docs = st.session_state.loader.load()
                
                # Validate documents were loaded
                if not st.session_state.docs:
                    st.error("❌ No PDFs found in ./us_census folder")
                    return False
                
                st.success(f"✅ Loaded {len(st.session_state.docs)} PDF(s)")
                
                # Split documents into chunks
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                st.session_state.final_documents = (
                    st.session_state.text_splitter.split_documents(
                        st.session_state.docs
                    )
                )
                
                st.success(
                    f"✅ Split into {len(st.session_state.final_documents)} chunks"
                )
                
                # Create vector store
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents,
                    st.session_state.embeddings
                )
                
                return True
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            return False
    
    return True


# ============================================================================
# SIDEBAR - SETTINGS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📁 Document Management")
    if st.button("🔄 Create/Refresh Vector Store from PDFs"):
        # Clear existing vectors to force reload
        if "vectors" in st.session_state:
            del st.session_state.vectors
        
        success = vector_embedding()
        if success:
            st.success("✅ Vector store created successfully!")
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.markdown(
        """
        **Model:** Gemma-7b-it  
        **Provider:** Groq  
        **Temperature:** 0.7
        
        **Technologies:**
        - Groq LLM for fast inference
        - FAISS for vector search
        - Streamlit for web UI
        - LangChain for orchestration
        """
    )


# ============================================================================
# MAIN CONTENT - Q&A INTERFACE
# ============================================================================

st.header("💬 Ask Questions About Your Documents")

# Check if vector store exists
if "vectors" not in st.session_state:
    st.info(
        "📌 **Get Started:**\n\n"
        "1. Place your PDF files in the `./us_census` folder\n"
        "2. Click 'Create/Refresh Vector Store' in the sidebar\n"
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
            "⚠️ Please create the vector store first by clicking "
            "'Create/Refresh Vector Store' in the sidebar."
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
