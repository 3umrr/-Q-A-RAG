
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


# ─── Backend health check ───────────────────────────────────────────────────────
@st.cache_data(ttl=60)
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


# ─── Auth Session State ─────────────────────────────────────────────────────────
if "token"    not in st.session_state: st.session_state.token    = None
if "username" not in st.session_state: st.session_state.username = None
if "user_id"  not in st.session_state: st.session_state.user_id  = None


def auth_headers() -> dict:
    """Return the Authorization header for all protected API calls."""
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ─── Login / Register Page ─────────────────────────────────────────────────────
def show_auth_page():
    """Display the login and registration forms side by side."""

    # Centered header
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            """
            <div style='text-align:center; padding: 2rem 0 1rem 0;'>
                <h1 style='font-size:2.5rem;'>📄 Q&A RAG</h1>
                <p style='color:gray; font-size:1.1rem;'>
                    Upload PDFs and ask questions using AI
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

   
    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    # ── Login Tab ──────────────────────────────────────────────────────────────
    with tab_login:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("### Welcome back!")
            username = st.text_input("Username", key="login_username", placeholder="your_username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")

            if st.button("Login →", use_container_width=True, type="primary", key="login_btn"):
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    try:
                        r = httpx.post(
                            f"{API_URL}/auth/login",
                            json={"username": username, "password": password},
                            timeout=10.0,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.session_state.token    = data["token"]
                            st.session_state.username = data["username"]
                            st.session_state.user_id  = data["user_id"]
                            st.success(data["message"])
                            st.rerun()
                        else:
                            try:
                                detail = r.json().get("detail", "Login failed")
                            except Exception:
                                detail = r.text or f"Server error {r.status_code}"
                            st.error(detail)
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    # ── Register Tab ───────────────────────────────────────────────────────────
    with tab_register:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("### Create your account")
            reg_username = st.text_input("Username", key="reg_username", placeholder="choose_a_username")
            reg_email    = st.text_input("Email",    key="reg_email",    placeholder="you@example.com")
            reg_password = st.text_input("Password", type="password", key="reg_password", placeholder="••••••••")
            reg_confirm  = st.text_input("Confirm password", type="password", key="reg_confirm", placeholder="••••••••")

            if st.button("Create account →", use_container_width=True, type="primary", key="register_btn"):
                if not all([reg_username, reg_email, reg_password, reg_confirm]):
                    st.error("Please fill in all fields.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        r = httpx.post(
                            f"{API_URL}/auth/register",
                            json={
                                "username": reg_username,
                                "email":    reg_email,
                                "password": reg_password,
                            },
                            timeout=10.0,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.session_state.token    = data["token"]
                            st.session_state.username = data["username"]
                            st.session_state.user_id  = data["user_id"]
                            st.success(data["message"])
                            st.rerun()
                        else:
                            try:
                                detail = r.json().get("detail", "Registration failed")
                            except Exception:
                                detail = r.text or f"Server error {r.status_code}"
                            st.error(detail)
                    except Exception as e:
                        st.error(f"Connection error: {e}")


# ─── Show auth page if not logged in ──────────────────────────────────────────
if not st.session_state.token:
    show_auth_page()
    st.stop()


# ─── From here on: user is authenticated ──────────────────────────────────────

# RAG session state (only initialised after login)
if "messages"       not in st.session_state: st.session_state.messages       = []
if "doc_sessions"   not in st.session_state: st.session_state.doc_sessions   = {}
if "active_session" not in st.session_state: st.session_state.active_session = None
if "upload_key"     not in st.session_state: st.session_state.upload_key     = 0


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
        with st.spinner(f"📤 Uploading and processing {session_name}..."):
            files_payload = [
                ("files", (f.name, f.read(), "application/pdf"))
                for f in pdf_files
            ]
            r = httpx.post(
                f"{API_URL}/upload",
                files=files_payload,
                headers=auth_headers(),
                timeout=120.0,
            )

        if r.status_code == 200:
            data       = r.json()
            session_id = data["session_id"]
            st.session_state.doc_sessions[session_id] = {
                "name":        session_name,
                "session_id":  session_id,
                "files":       data["files"],
                "doc_count":   data["doc_count"],
                "chunk_count": data["chunk_count"],
            }
            st.session_state.active_session = session_id
            st.session_state.messages       = []
            st.success(data["message"])
            return True
        else:
            st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
            return False

    except httpx.TimeoutException:
        st.error("❌ Upload timed out. The PDF may be too large.")
        return False
    except Exception as e:
        st.error(f"❌ {e}")
        return False


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logged-in user info
    st.markdown(f"👤 **{st.session_state.username}**")
    if st.button("Logout", key="logout_btn"):
        for key in ["token", "username", "user_id", "messages", "doc_sessions", "active_session"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.header("📂 Document Sessions")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.upload_key}",
    )

    if uploaded_files and st.button("🚀 Process PDFs", use_container_width=True):
        if process_pdfs(uploaded_files):
            st.session_state.upload_key += 1  # resets the file uploader widget
        st.rerun()

    st.divider()

    if st.session_state.doc_sessions:
        st.subheader("Active Sessions")
        for sid, meta in list(st.session_state.doc_sessions.items()):
            is_active = sid == st.session_state.active_session
            label     = f"{'✅ ' if is_active else ''}📄 {meta['name']}"
            if st.button(label, key=f"session_{sid}", use_container_width=True):
                st.session_state.active_session = sid
                st.session_state.messages       = []
                st.rerun()

            st.caption(
                f"📄 {meta['doc_count']} pages · "
                f"🧩 {meta['chunk_count']} chunks · "
                f"`{meta['session_id'][:8]}...`"
            )

            if st.button("🗑️ Remove session", key=f"remove_{sid}"):
                try:
                    httpx.delete(
                        f"{API_URL}/sessions/{sid}",
                        headers=auth_headers(),
                        timeout=5.0,
                    )
                except Exception:
                    pass
                st.session_state.doc_sessions.pop(sid, None)
                if st.session_state.active_session == sid:
                    st.session_state.active_session = None
                    st.session_state.messages       = []
                st.rerun()
    else:
        st.info("Upload a PDF to get started.")


# ─── Main Chat Area ────────────────────────────────────────────────────────────
st.title("💬 Q&A RAG")

if not st.session_state.active_session:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

active_meta = st.session_state.doc_sessions.get(st.session_state.active_session)
if active_meta:
    st.caption(
        f"Active document: **{active_meta['name']}** · "
        f"{active_meta['doc_count']} pages · "
        f"{active_meta['chunk_count']} chunks"
    )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_question = st.chat_input("Ask a question about your document...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    api_session_id = st.session_state.active_session

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        start_time    = time.time()

        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/ask",
                json={
                    "session_id":   api_session_id,
                    "question":     user_question,
                    "chat_history": format_chat_history(),
                },
                headers=auth_headers(),
                timeout=120.0,
            ) as response:
                for chunk in response.iter_text():
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)
            elapsed = time.time() - start_time
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            col1, col2 = st.columns(2)
            with col1: st.caption(f"⏱️ {elapsed:.2f}s")
            with col2: st.caption(f"📝 {len(full_response)} chars")

            with st.expander("📄 View Retrieved Context"):
                try:
                    ctx_r = httpx.get(
                        f"{API_URL}/sessions/{api_session_id}/context",
                        params={"question": user_question},
                        headers=auth_headers(),
                        timeout=15.0,
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
    unsafe_allow_html=True,
)
