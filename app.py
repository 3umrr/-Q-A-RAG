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


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="📄 Q&A with PDFs",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


def validate_api_keys():
    if not groq_api_key:
        st.error(
            "❌ **GROQ_API_KEY not found!**\n\n"
            "Please add your Groq API key to the `.env` file:\n"
            "`GROQ_API_KEY=your_key_here`"
        )
        st.stop()


validate_api_keys()


@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )


llm = load_llm()

st.title("📄 Q&A with your PDFs")
st.markdown("Ask questions about your PDF documents and get AI-powered answers")


qa_prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    Context: {context}
    
    Question: {input}
    
    Answer:"""
)


def vector_embedding(pdf_files):
    if not pdf_files:
        st.error("❌ No PDF files provided")
        return False
    
    try:
        with st.spinner("📥 Loading and processing PDFs..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            st.session_state.docs = []
            
            for pdf_file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    st.session_state.docs.extend(docs)
                finally:
                    os.unlink(tmp_path)
            
            if not st.session_state.docs:
                st.error("❌ No content found in the uploaded PDFs")
                return False
            
            st.success(f"✅ Loaded {len(st.session_state.docs)} pages from PDF(s)")
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(
                    st.session_state.docs
                )
            )
            
            if not st.session_state.final_documents:
                st.warning("⚠️ PDF content is too small to split into chunks. Using original documents...")
                st.session_state.final_documents = st.session_state.docs
            
            st.success(
                f"✅ Split into {len(st.session_state.final_documents)} chunks"
            )
            
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            
            st.session_state.pdf_uploaded = True
            return True
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return False


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
            if "vectors" in st.session_state:
                del st.session_state.vectors
            
            success = vector_embedding(uploaded_files)
            if success:
                st.success("✅ PDFs processed and ready for Q&A!")
    else:
        st.info("👆 Upload PDF files to get started")


st.header("💬 Ask Questions About Your Documents")

if "vectors" not in st.session_state:
    st.info(
        "📌 **Get Started:**\n\n"
        "1. Upload your PDF files using the uploader in the sidebar\n"
        "2. Click 'Process PDFs'\n"
        "3. Then ask your questions here!"
    )
else:
    st.success("✅ Vector store loaded and ready!")

st.subheader("Enter Your Question")
user_question = st.text_input(
    label="Your question:",
    placeholder="What information is in the documents?",
    label_visibility="collapsed"
)

if user_question:
    if "vectors" not in st.session_state:
        st.warning(
            "⚠️ Please upload and process PDFs first using the "
            "file uploader in the sidebar."
        )
    else:
        try:
            with st.spinner("🤔 Searching documents and generating answer..."):
                retriever = st.session_state.vectors.as_retriever()
                
                def format_docs(docs):
                    return "\n\n".join(
                        f"Document {i+1}:\n{doc.page_content}"
                        for i, doc in enumerate(docs)
                    )
                
                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "input": RunnablePassthrough()
                    }
                    | qa_prompt
                    | llm
                    | StrOutputParser()
                )
                
                start_time = time.time()
                
                response = rag_chain.invoke(user_question)
                
                elapsed_time = time.time() - start_time
            
            st.subheader("✅ Answer")
            st.markdown(response)
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
            with col2:
                st.caption(f"📝 Response length: {len(response)} characters")
        
        except Exception as error:
            st.error(f"❌ Error processing question: {str(error)}")


st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    📄 Q&A RAG Application | Made with ❤️ using Streamlit & Groq
    </div>
    """,
    unsafe_allow_html=True
)
