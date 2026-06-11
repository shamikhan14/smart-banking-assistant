"""
NorthStar Bank — Smart Banking Assistant UI
Run from project root:
    streamlit run app.py
"""

import os
import tempfile
import pathlib
import time
import json
import requests
import re
import html as _html_lib

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Client-side HTML sanitizer ────────────────────────────────────────────────
_STRIP_HTML_RE = re.compile(r"<[^>]+>", re.DOTALL)

def _sanitize_answer(text: str) -> str:
    """Remove HTML tags from answer text; preserve the plain content."""
    return _STRIP_HTML_RE.sub("", text or "").strip()

# ── Streaming API base URL ────────────────────────────────────────────────────
_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NorthStar Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0f1e; color: #e8eaf0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.ns-header {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5e 50%, #0d1b3e 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.ns-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.ns-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #f0f4ff;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.ns-header p { color: #7b9cc4; margin: 0; font-size: 0.9rem; font-weight: 300; }
.ns-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1626;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1a2a45;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #7b9cc4;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.5rem 1.25rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1a3a6e !important;
    color: #e8f0fe !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Chat bubbles ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #1e3a6e, #1a2f5e);
    border: 1px solid #2a4a80;
    border-radius: 16px 16px 4px 16px;
    padding: 0.875rem 1.125rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: #e8f0fe;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-bubble-assistant {
    background: #0d1626;
    border: 1px solid #1a2a45;
    border-radius: 16px 16px 16px 4px;
    padding: 0.875rem 1.125rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #c8d8f0;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.label-user      { color: #60a5fa; }
.label-assistant { color: #34d399; }

/* ── Source info text (replaces old citation-card HTML) ── */
.source-line {
    font-size: 0.75rem;
    color: #4a6a94;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #1a2a45;
}

/* ── SQL block ── */
.sql-block {
    background: #050d1a;
    border: 1px solid #1a2a45;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #67e8f9;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ── Input & buttons ── */
.stTextInput > div > div > input {
    background: #0d1626 !important;
    border: 1px solid #1a2a45 !important;
    border-radius: 12px !important;
    color: #e8f0fe !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #3d5070 !important; }

.stButton > button {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
}

.stFileUploader > div {
    background: #0d1626 !important;
    border: 2px dashed #1a2a45 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease !important;
}
.stFileUploader > div:hover { border-color: #3b82f6 !important; }

hr { border-color: #1a2a45 !important; }

.doc-item {
    background: #0d1626;
    border: 1px solid #1a2a45;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #7b9cc4;
}
.doc-name { color: #c8d8f0; font-weight: 500; }

/* ── Thinking indicator ── */
@keyframes thinking-pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.85); }
    50%       { opacity: 1;   transform: scale(1.1); }
}
.thinking-bar {
    background: #0d1626;
    border: 1px solid #1a2a45;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 3rem 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.thinking-dots {
    display: flex;
    gap: 5px;
    align-items: center;
}
.thinking-dots span {
    width: 8px; height: 8px;
    background: #3b82f6;
    border-radius: 50%;
    display: inline-block;
    animation: thinking-pulse 1.4s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; background: #60a5fa; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; background: #93c5fd; }
.thinking-text {
    color: #4a6a94;
    font-size: 0.82rem;
    font-style: italic;
    letter-spacing: 0.3px;
}

/* ── Disabled input while processing ── */
.stTextInput > div > div > input:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-header">
    <div class="ns-badge">Smart Banking Assistant</div>
    <h1>🏦 NorthStar Bank</h1>
    <p>Ask about products, policies, account details, or upload new documents to the knowledge base.</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_upload = st.tabs(["💬  Chat", "📄  Upload Documents"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # ── Input row — always rendered first so it never disappears ─────────────
    col_input, col_send, col_clear = st.columns([8, 1, 1])

    with col_input:
        user_input = st.text_input(
            label="query",
            placeholder="Ask anything about NorthStar Bank...",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.input_key}",
        )

    with col_send:
        send_clicked = st.button(
            "Send",
            use_container_width=True,
            key="btn_send",
        )

    with col_clear:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True,
            key="btn_clear",
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Button handlers ───────────────────────────────────────────────────────
    if clear_clicked:
        st.session_state.messages = []
        st.session_state.pending_query = ""
        st.session_state.input_key += 1
        st.rerun()

    if send_clicked and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.pending_query = user_input.strip()
        st.session_state.input_key += 1
        st.rerun()

    # ── Process any pending query ─────────────────────────────────────────────
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = ""

        # Build chat_history from all messages except the just-appended user msg
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "assistant", "content": msg.get("answer", "")})

        # ── Thinking indicator placeholder ────────────────────────────────────
        stream_placeholder = st.empty()
        stream_placeholder.markdown(
            """<div>
                <div class="chat-label label-assistant">NorthStar Assistant</div>
                <div class="thinking-bar">
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <div class="thinking-text">Thinking…</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        full_answer = ""
        final_meta = {
            "policy_citations": "",
            "page_no": "",
            "document_name": "",
            "sql_query_executed": None,
        }

        try:
            with requests.post(
                f"{_API_BASE}/api/v1/query/stream",
                json={"query": query, "chat_history": chat_history},
                stream=True,
                timeout=120,
                headers={"Accept": "text/event-stream"},
            ) as resp:

                # ── Input guardrail blocked (HTTP 400) ────────────────────
                if resp.status_code == 400:
                    try:
                        detail   = resp.json().get("detail", {})
                        guard_msg = detail.get("message", "Your message was blocked by a safety filter.")
                    except Exception:
                        guard_msg = "Your message was blocked by a safety filter."
                    stream_placeholder.empty()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "answer": f"🚫 **Guardrail triggered:** {guard_msg}",
                        "policy_citations": "",
                        "page_no": "",
                        "document_name": "",
                        "sql_query_executed": None,
                    })
                    st.rerun()

                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):]

                    if payload_str.strip() == "[DONE]":
                        break

                    try:
                        payload = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    # ── Output guardrail blocked (SSE event) ──────────────
                    if payload.get("guardrail_error"):
                        stream_placeholder.empty()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "answer": f"🚫 **Output blocked:** {payload.get('message', 'Response flagged by safety filter.')}",
                            "policy_citations": "",
                            "page_no": "",
                            "document_name": "",
                            "sql_query_executed": None,
                        })
                        st.rerun()

                    if "token" in payload:
                        full_answer += payload["token"]
                        safe_token = _sanitize_answer(full_answer)
                        stream_placeholder.markdown(
                            f"""<div>
                                <div class="chat-label label-assistant">NorthStar Assistant</div>
                                <div class="chat-bubble-assistant">{safe_token}▌</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    for field in ("policy_citations", "page_no", "document_name", "sql_query_executed"):
                        if field in payload and "token" not in payload:
                            final_meta[field] = payload[field]

            stream_placeholder.empty()

            st.session_state.messages.append({
                "role": "assistant",
                "answer": _sanitize_answer(full_answer) or "No answer returned.",
                **final_meta,
            })

        except requests.HTTPError as e:
            stream_placeholder.empty()
            try:
                detail    = e.response.json().get("detail", {})
                guard_msg = detail.get("message") or str(e)
            except Exception:
                guard_msg = str(e)
            st.session_state.messages.append({
                "role": "assistant",
                "answer": f"🚫 **Request blocked:** {guard_msg}",
                "policy_citations": "",
                "page_no": "",
                "document_name": "",
                "sql_query_executed": None,
            })

        except Exception as e:
            stream_placeholder.empty()
            st.session_state.messages.append({
                "role": "assistant",
                "answer": f"⚠️ An error occurred: {str(e)}",
                "policy_citations": "",
                "page_no": "",
                "document_name": "",
                "sql_query_executed": None,
            })

    # ── Chat history display — newest first ───────────────────────────────────
    # Pair up messages (user + assistant) and reverse so newest pair is at top.
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #3d5070;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">💬</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#7b9cc4; margin-bottom:0.5rem;">
                Start a conversation
            </div>
            <div style="font-size:0.82rem; line-height:1.8;">
                Try: <em>"What is the minimum CIBIL score for personal loans?"</em><br>
                Or: <em>"Show me transactions for account 1345367"</em><br>
                Or: <em>"What are the credit card interest rates?"</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Build (user_msg, assistant_msg) pairs ─────────────────────────────
        # Walk the flat list and group consecutive user→assistant turns.
        # Unpaired messages (e.g. a user msg still awaiting a response) are
        # shown on their own.
        pairs: list[tuple] = []
        i = 0
        msgs = st.session_state.messages
        while i < len(msgs):
            if msgs[i]["role"] == "user":
                user_msg = msgs[i]
                # Check if the next message is the assistant's reply
                if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                    pairs.append((user_msg, msgs[i + 1]))
                    i += 2
                else:
                    pairs.append((user_msg, None))
                    i += 1
            else:
                # Orphaned assistant message (shouldn't normally happen)
                pairs.append((None, msgs[i]))
                i += 1

        # Reverse so newest conversation turn appears at the top
        for user_msg, assistant_msg in reversed(pairs):

            # ── Assistant bubble ──────────────────────────────────────────────
            if assistant_msg:
                answer   = assistant_msg.get("answer", "")
                citation = assistant_msg.get("policy_citations", "")
                page     = assistant_msg.get("page_no", "")
                doc      = assistant_msg.get("document_name", "")
                sql      = assistant_msg.get("sql_query_executed")

                # Build plain-text source line (no HTML cards)
                source_parts = []
                if doc and doc not in ("N/A", "", "agentic_rag_db"):
                    source_parts.append(f"📄 {doc}")
                if page and page not in ("N/A", ""):
                    source_parts.append(f"Page {page}")
                if citation and citation not in ("N/A", ""):
                    source_parts.append(citation)

                source_line = "  ·  ".join(source_parts) if source_parts else ""

                # SQL block (kept as preformatted text, not HTML)
                sql_section = f"\n\n🔍 SQL:\n{sql}" if sql else ""

                # Render assistant bubble — answer is plain text; source is
                # a small muted line appended below, SQL in a code block
                st.markdown(
                    f'<div class="chat-label label-assistant">NorthStar Assistant</div>',
                    unsafe_allow_html=True,
                )
                with st.container():
                    st.markdown(
                        f'<div class="chat-bubble-assistant">{answer}</div>',
                        unsafe_allow_html=True,
                    )
                    if source_line:
                        st.markdown(
                            f'<div class="source-line">📎 {source_line}</div>',
                            unsafe_allow_html=True,
                        )
                    if sql:
                        st.code(sql, language="sql")

            # ── User bubble ───────────────────────────────────────────────────
            if user_msg:
                st.markdown(
                    f"""<div class="chat-label label-user">You</div>
                    <div class="chat-bubble-user">{user_msg["content"]}</div>""",
                    unsafe_allow_html=True,
                )

            # Subtle divider between conversation turns
            st.markdown("<hr>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:

    col_upload, col_docs = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div style="margin-bottom:1.25rem;">
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#e8f0fe; margin-bottom:0.4rem;">
                Upload PDF to Knowledge Base
            </div>
            <div style="font-size:0.82rem; color:#7b9cc4; line-height:1.6;">
                Uploaded PDFs are parsed, chunked, embedded, and stored in the vector database.
                The assistant will use them to answer future questions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        if uploaded_file is not None:
            st.markdown(f"""
            <div class="doc-item">
                <span>📄</span>&nbsp;
                <div>
                    <div class="doc-name">{uploaded_file.name}</div>
                    <div style="font-size:0.75rem;">{uploaded_file.size / 1024:.1f} KB · Ready to ingest</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            ingest_clicked = st.button("⚡  Ingest Document", use_container_width=True, key="btn_ingest")

            if ingest_clicked:
                progress_bar = st.progress(0, text="Saving file...")
                status = st.empty()

                try:
                    suffix = pathlib.Path(uploaded_file.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    progress_bar.progress(15, text="Uploading to backend...")
                    status.info("🔍 Sending PDF to the ingestion service...")

                    # ── Send to backend API (avoids psycopg_pool / DB imports in UI) ──
                    with open(tmp_path, "rb") as pdf_file:
                        ingest_resp = requests.post(
                            f"{_API_BASE}/api/v1/ingest",
                            files={"file": (uploaded_file.name, pdf_file, "application/pdf")},
                            timeout=300,
                        )

                    progress_bar.progress(80, text="Processing chunks...")

                    if ingest_resp.status_code == 200:
                        result = ingest_resp.json()
                        progress_bar.progress(100, text="Done!")

                        st.session_state.ingested_docs.append({
                            "name": uploaded_file.name,
                            "chunks": result.get("chunks_ingested", 0),
                            "doc_id": result.get("doc_id", ""),
                        })

                        status.success(
                            f"✅ **{uploaded_file.name}** ingested — "
                            f"{result.get('chunks_ingested', 0)} chunks stored."
                        )
                    else:
                        try:
                            err_detail = ingest_resp.json().get("detail", ingest_resp.text)
                        except Exception:
                            err_detail = ingest_resp.text
                        progress_bar.empty()
                        status.error(f"❌ Ingestion failed (HTTP {ingest_resp.status_code}): {err_detail}")

                    pathlib.Path(tmp_path).unlink(missing_ok=True)

                except requests.exceptions.ConnectionError:
                    progress_bar.empty()
                    status.error(
                        "❌ Could not connect to the backend. "
                        "Make sure the FastAPI server is running (`uvicorn main:app --reload`) "
                        "and `API_BASE_URL` in your `.env` is correct."
                    )
                except Exception as e:
                    progress_bar.empty()
                    status.error(f"❌ Ingestion failed: {str(e)}")

        else:
            st.markdown("""
            <div style="text-align:center; padding:2rem 1rem; color:#3d5070;
                        border:1px dashed #1a2a45; border-radius:12px;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">📂</div>
                <div style="font-size:0.85rem;">Drag & drop a PDF or click Browse files</div>
            </div>
            """, unsafe_allow_html=True)

    with col_docs:
        st.markdown("""
        <div style="margin-bottom:1.25rem;">
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#e8f0fe; margin-bottom:0.4rem;">
                Ingested This Session
            </div>
            <div style="font-size:0.82rem; color:#7b9cc4; line-height:1.6;">
                Documents processed and added to the knowledge base.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.ingested_docs:
            st.markdown("""
            <div style="text-align:center; padding:2.5rem 1rem; color:#3d5070;
                        border:1px dashed #1a2a45; border-radius:12px;">
                <div style="font-size:1.8rem; margin-bottom:0.75rem;">📭</div>
                <div style="font-size:0.85rem;">No documents ingested yet this session.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for doc in st.session_state.ingested_docs:
                st.markdown(f"""
                <div class="doc-item">
                    ✅&nbsp;
                    <div>
                        <div class="doc-name">{doc['name']}</div>
                        <div style="font-size:0.75rem;">
                            {doc['chunks']} chunks &nbsp;·&nbsp; ID: {doc['doc_id'][:8]}…
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.18);
                    border-radius:10px; padding:1rem 1.125rem; font-size:0.82rem;
                    color:#7b9cc4; line-height:1.7;">
            <strong style="color:#93c5fd;">💡 Tips</strong><br>
            • PDFs with selectable text ingest faster and more accurately.<br>
            • Scanned PDFs use OCR — may take longer.<br>
            • Images and charts are described via GPT-4o vision.<br>
            • Re-uploading the same filename replaces old chunks.
        </div>
        """, unsafe_allow_html=True)