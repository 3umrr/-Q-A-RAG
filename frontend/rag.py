import os
import tempfile

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder


# ─── LLM ──────────────────────────────────────────────────────────────────────

def load_llm(api_key: str) -> ChatGroq:
    """Initialize and return the Groq LLM."""
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )


# ─── Embeddings ────────────────────────────────────────────────────────────────

def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─── Prompt ────────────────────────────────────────────────────────────────────

def get_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """You are a helpful assistant that answers questions about documents.

        Use the context below to answer the question.
        Start your answer directly — do not list or repeat content from the context as a preamble.
        Be concise and answer in your own words.
        Do not copy 'Question:' / 'Answer:' pairs from the context — those are document content.
        Use the chat history to understand follow-up questions.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question: {input}

        Answer:"""
    )


# ─── Document Loading ──────────────────────────────────────────────────────────

def load_pdfs(pdf_files: list) -> list:
    """
    Load a list of uploaded PDF file objects and return LangChain documents.
    Writes each PDF to a temp file, loads it, then cleans up.
    """
    all_docs = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)
    return all_docs


# ─── Text Splitting ────────────────────────────────────────────────────────────

def split_documents(docs: list, embeddings: HuggingFaceEmbeddings) -> list:
   
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80   
    )
    chunks = chunker.split_documents(docs)

    return chunks if chunks else docs


# ─── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Build and return a FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


# ─── Hybrid Retriever ────────────────────────────────────────────────────

def build_hybrid_retriever(chunks: list, vector_store: FAISS, k: int = 6):
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )


# ─── Reranker ─────────────────────────────────────────────────────────────────

def build_reranker(base_retriever, top_n: int = 3):
   
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(query: str) -> list:
        candidates = base_retriever.invoke(query)
        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]

        scores = model.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    
    return RunnableLambda(rerank)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
   
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ─── RAG Chain ─────────────────────────────────────────────────────────────────

def get_rag_chain(retriever, llm: ChatGroq, prompt: ChatPromptTemplate, chat_history: str = ""):
    
    return (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda _: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
