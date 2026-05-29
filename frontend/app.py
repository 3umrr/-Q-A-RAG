

import os
import time
import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


API_URL = os.getenv("API_URL", "http://localhost:8000")


# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Q&A RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data(ttl=5)   
def check_backend() -> bool:
    try:
        r = httpx.get(f"{API_URL}/", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

if not check_backend():
    st.error(
        "❌ **FastAPI backend is not running.**\n\n"
        "Start it with:\n"
        "```\npython -m uvicorn backend.main:app --port 8000\n```"
    )
    st.stop()



if "messages"       not in st.session_state: st.session_state.messages       = []
if "doc_sessions"   not in st.session_state: st.session_state.doc_sessions   = {}
if "active_session" not in st.session_state: st.session_state.active_session = None


# ─── Chat History Formatter ────────────────────────────────────────────────────
def format_chat_history() -> str:
    recent = st.session_state.messages[-10:]
    if not recent:
        return "No previous conversation."
    return "\n".join(
        ("Human" if m["role"] == "user" else "Assistant") + ": " + m["content"]
        for m in recent
    )


# ─── PDF Processing via API ────────────────────────────────────────────────────
def process_pdfs(pdf_files) -> bool:
    
    session_name = ", ".join(f.name for f in pdf_files)

    try:
        with st.spinner("Uploading and processing PDFs..."):
           
            files_payload = [
                ("files", (f.name, f.read(), "application/pdf"))
                for f in pdf_files
            ]

            r = httpx.post(
                f"{API_URL}/upload",
                files=files_payload,
                timeout=180.0   # chunking + reranker can take time
            )

        if r.status_code != 200:
            st.error(f"❌ Upload failed: {r.json().get('detail', r.text)}")
            return False

        data = r.json()

        # Store only lightweight metadata — NOT the retriever object
        st.session_state.doc_sessions[session_name] = {
            "session_id":  data["session_id"],   # UUID from the API
            "doc_count":   data["doc_count"],
            "chunk_count": data["chunk_count"],
        }
        st.session_state.active_session = session_name
        return True

    except httpx.ConnectError:
        st.error("❌ Cannot connect to the backend. Is it running?")
        return False
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Backend status indicator ───────────────────────────────────────────────
    st.caption(f"🟢 Backend: `{API_URL}`")

    # ── Upload PDFs ────────────────────────────────────────────────────────────
    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    if uploaded_files:
        if st.button("🚀 Process PDFs", key="process_btn", use_container_width=True):
            ok = process_pdfs(uploaded_files)
            if ok:
                st.success("✅ Ready!")
    else:
        st.info("👆 Upload PDF files to get started")

    # ── Document Sessions ──────────────────────────────────────────────────────
    if st.session_state.doc_sessions:
        st.divider()
        st.subheader("📂 Document Sessions")

        session_names = list(st.session_state.doc_sessions.keys())
        if st.session_state.active_session not in session_names:
            st.session_state.active_session = session_names[0]

        selected = st.radio(
            "Active document set:",
            session_names,
            index=session_names.index(st.session_state.active_session),
            key="session_radio"
        )
        st.session_state.active_session = selected

        meta = st.session_state.doc_sessions[selected]
        st.caption(
            f"📄 {meta['doc_count']} pages · "
            f"🧩 {meta['chunk_count']} chunks · "
            f"`{meta['session_id'][:8]}...`"   
        )

        if st.button("🗑️ Remove session", key="remove_session"):
           
            api_sid = meta["session_id"]
            try:
                httpx.delete(f"{API_URL}/sessions/{api_sid}", timeout=5.0)
            except Exception:
                pass   # best-effort
            del st.session_state.doc_sessions[selected]
            remaining = list(st.session_state.doc_sessions.keys())
            st.session_state.active_session = remaining[0] if remaining else None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📄 Q&A with your PDFs")

if st.session_state.active_session:
    st.success(f"📂 Active: **{st.session_state.active_session}**")
else:
    st.info(
        "📌 **Get Started:**\n\n"
        "1. Upload your PDF files in the sidebar\n"
        "2. Click **Process PDFs**\n"
        "3. Ask your questions below!"
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_question = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.active_session is None
)

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Get the API session_id for the active document set
    active_meta = st.session_state.doc_sessions[st.session_state.active_session]
    api_session_id = active_meta["session_id"]

    try:
        with st.chat_message("assistant"):
            placeholder   = st.empty()
            full_response = ""
            start_time    = time.time()

            
            with httpx.stream(
                "POST",
                f"{API_URL}/ask",
                json={
                    "session_id":   api_session_id,
                    "question":     user_question,
                    "chat_history": format_chat_history(),
                },
                timeout=60.0
            ) as r:
                for chunk in r.iter_text():
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            elapsed = time.time() - start_time

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # ── Stats ──────────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1: st.caption(f"⏱️ {elapsed:.2f}s")
        with col2: st.caption(f"📝 {len(full_response)} chars")

      
        with st.expander("📄 View Retrieved Context"):
            try:
                ctx_r = httpx.get(
                    f"{API_URL}/sessions/{api_session_id}/context",
                    params={"question": user_question},
                    timeout=15.0
                )
                if ctx_r.status_code == 200:
                    chunks = ctx_r.json()["chunks"]
                    for c in chunks:
                        page   = f"Page {c['page'] + 1}" if c["page"] is not None else ""
                        source = os.path.basename(c["source"]) if c["source"] else ""
                        label  = " · ".join(filter(None, [f"Chunk {c['index']}", page, source]))
                        st.markdown(f"**{label}**")
                        st.write(c["content"])
                        st.divider()
            except Exception:
                st.caption("Context unavailable")

    except httpx.ReadTimeout:
        st.error("❌ Request timed out. The model may be busy.")
    except Exception as error:
        st.error(f"❌ {error}")


# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.8em'>"
    "📄 Q&A RAG · Streamlit + FastAPI + Groq</div>",
    unsafe_allow_html=True
)
