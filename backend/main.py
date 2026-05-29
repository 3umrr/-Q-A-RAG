

import os
import sys
import uuid
import tempfile
from contextlib import asynccontextmanager

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
import rag
import backend.database as db



sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────────────────
    print("[STARTUP] Loading models...")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    app.state.llm        = rag.load_llm(groq_api_key)
    app.state.embeddings = rag.load_embeddings()
    app.state.qa_prompt  = rag.get_qa_prompt()


    db.create_tables()

   
    persisted = db.get_all_sessions()
    for row in persisted:
        sessions[row["id"]] = {
            "files":       row["files"],
            "doc_count":   row["doc_count"],
            "chunk_count": row["chunk_count"],
            "retriever":   None,   
        }
    print(f"[STARTUP] Restored {len(persisted)} session(s) from database.")
    print("[STARTUP] Models loaded. Server ready.")
    yield
    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    print("[SHUTDOWN] Clearing in-memory sessions.")
    sessions.clear()


# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Q&A RAG API",
    description="Upload PDFs and ask questions using RAG + Groq.",
    version="2.0.0",
    lifespan=lifespan,   
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class UploadResponse(BaseModel):
    session_id:  str        
    files:       list[str]  
    doc_count:   int        
    chunk_count: int        
    message:     str        
class SessionInfo(BaseModel):
    
    session_id:  str
    files:       list[str]
    doc_count:   int
    chunk_count: int


# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — confirms the API is running."""
    return {
        "status":   "ok",
        "service":  "Q&A RAG API",
        "version":  "2.0.0",
        "sessions": len(sessions),
    }



@app.post("/upload", response_model=UploadResponse)
def upload_pdfs(files: list[UploadFile] = File(...)):
   
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

 
    temp_paths = []
    filenames  = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            if not upload.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{upload.filename}' is not a PDF"
                )
          
            content = upload.file.read()          
          
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                f.write(content)

            temp_paths.append(dest)
            filenames.append(upload.filename)

        
        from langchain_community.document_loaders import PyPDFLoader

        docs = []
        for path in temp_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        if not docs:
            raise HTTPException(status_code=422, detail="No text found in PDFs")

        embeddings = app.state.embeddings

        chunks           = rag.split_documents(docs, embeddings)
        vectors          = rag.build_vector_store(chunks, embeddings)
        hybrid_retriever = rag.build_hybrid_retriever(chunks, vectors)
        retriever        = rag.build_reranker(hybrid_retriever, top_n=4)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "retriever":   retriever,
        "files":       filenames,
        "doc_count":   len(docs),
        "chunk_count": len(chunks),
    }

  
    db.save_session(
        session_id  = session_id,
        files       = filenames,
        doc_count   = len(docs),
        chunk_count = len(chunks),
    )

    return UploadResponse(
        session_id  = session_id,
        files       = filenames,
        doc_count   = len(docs),
        chunk_count = len(chunks),
        message     = f"✅ {len(files)} file(s) processed. Use session_id to ask questions.",
    )


@app.get("/sessions", response_model=list[SessionInfo])
def list_sessions():
    """List all active document sessions."""
    return [
        SessionInfo(
            session_id  = sid,
            files       = data["files"],
            doc_count   = data["doc_count"],
            chunk_count = data["chunk_count"],
        )
        for sid, data in sessions.items()
    ]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
  
    if session_id not in sessions:
        
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    del sessions[session_id]
    db.delete_session_db(session_id)
    return {"message": f"Session '{session_id}' deleted"}



class AskRequest(BaseModel):
    session_id:   str
    question:     str
    chat_history: str = "No previous conversation."  


class AskResponse(BaseModel):
    """Used only for the non-streaming version."""
    answer:     str
    session_id: str
    chunks_used: int



@app.post("/ask")
def ask_streaming(request: AskRequest):
   
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Upload a PDF first."
        )

    session   = sessions[request.session_id]
    retriever = session["retriever"]

  
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Session '{request.session_id}' was restored from the database "
                "but the document index was lost on server restart. "
                "Please re-upload your PDF to rebuild the index."
            )
        )

    llm       = app.state.llm
    qa_prompt = app.state.qa_prompt

 
    rag_chain = rag.get_rag_chain(
        retriever,
        llm,
        qa_prompt,
        chat_history=request.chat_history
    )


    def token_generator():
        full_answer = []
        try:
            for token in rag_chain.stream(request.question):
                full_answer.append(token)
                yield token
        except Exception as e:
            yield f"\n[ERROR] {str(e)}"
        finally:
          
            answer_text = "".join(full_answer)
            if answer_text:
                db.save_message(request.session_id, "user",      request.question)
                db.save_message(request.session_id, "assistant", answer_text)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        token_generator(),
        media_type="text/plain"  
    )


@app.post("/ask/sync", response_model=AskResponse)
def ask_sync(request: AskRequest):
    
   
    if request.session_id not in sessions:
        raise HTTPException(status_code=404,
                            detail=f"Session '{request.session_id}' not found")

    session   = sessions[request.session_id]
    retriever = session["retriever"]

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Session '{request.session_id}' was restored from the database "
                "but the document index was lost on server restart. "
                "Please re-upload your PDF to rebuild the index."
            )
        )

    rag_chain = rag.get_rag_chain(
        retriever,
        app.state.llm,
        app.state.qa_prompt,
        chat_history=request.chat_history
    )

    
    retrieved = retriever.invoke(request.question)
    answer    = rag_chain.invoke(request.question)


    db.save_message(request.session_id, "user",      request.question)
    db.save_message(request.session_id, "assistant", answer)

    return AskResponse(
        answer      = answer,
        session_id  = request.session_id,
        chunks_used = len(retrieved)
    )



@app.get("/sessions/{session_id}/context")
def get_context(session_id: str, question: str):
   
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    retriever = sessions[session_id]["retriever"]
    docs      = retriever.invoke(question)

    return {
        "question": question,
        "chunks": [
            {
                "index":   i + 1,
                "page":    doc.metadata.get("page", None),
                "source":  doc.metadata.get("source", ""),
                "content": doc.page_content[:300] + "..."
                           if len(doc.page_content) > 300 else doc.page_content
            }
            for i, doc in enumerate(docs)
        ]
    }


@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
   
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [
            {
                "role":       m["role"],
                "content":    m["content"],
                "created_at": str(m["created_at"])
            }
            for m in messages
        ]
    }
