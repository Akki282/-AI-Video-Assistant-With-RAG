"""
Streamlit UI for the AI Video Assistant pipeline.

Run with:
    streamlit run streamlit_app.py

This assumes the same project structure as your CLI script, i.e. it can
import:
    utils.audio_processor.process_input
    core.transcriber.transcribe_all
    core.summarizer.summarize, generate_title
    core.extractor.extract_action_items, extract_key_decisions, extract_questions
    core.rag_engine.build_rag_chain, ask_question
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "processing" not in st.session_state:
    st.session_state.processing = False


def run_pipeline(source: str, language: str) -> dict:
    """Same logic as the CLI run_pipeline, wrapped for Streamlit progress UI."""
    progress = st.progress(0, text="Starting AI Video Assistant…")

    progress.progress(10, text="Processing input source…")
    chunks = process_input(source)

    progress.progress(30, text="Transcribing audio…")
    transcript = transcribe_all(chunks, language=language)

    progress.progress(55, text="Generating title…")
    title = generate_title(transcript)

    progress.progress(65, text="Summarizing…")
    summary = summarize(transcript)

    progress.progress(75, text="Extracting action items…")
    action_items = extract_action_items(transcript)

    progress.progress(85, text="Extracting key decisions…")
    decisions = extract_key_decisions(transcript)

    progress.progress(90, text="Extracting open questions…")
    questions = extract_questions(transcript)

    progress.progress(97, text="Building chat engine…")
    rag_chain = build_rag_chain(transcript)

    progress.progress(100, text="Done!")
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------
# Sidebar — input controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 AI Video Assistant")
    st.caption("Transcribe, summarize, and chat with any video or audio.")

    input_mode = st.radio("Source type", ["YouTube URL", "Upload file"], horizontal=False)

    source = None
    uploaded_path = None

    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm"],
        )
        if uploaded_file is not None:
            tmp_dir = tempfile.gettempdir()
            uploaded_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(uploaded_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = uploaded_path

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_clicked = st.button(
        "🚀 Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing or not source,
    )

    if st.session_state.result is not None:
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------------------------------
# Run pipeline on click
# --------------------------------------------------------------------------
if run_clicked and source:
    st.session_state.processing = True
    st.session_state.chat_history = []
    try:
        with st.spinner("Working on it…"):
            st.session_state.result = run_pipeline(source, language)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.session_state.result = None
    finally:
        st.session_state.processing = False

# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.info("👈 Provide a YouTube URL or upload a file, then click **Run Pipeline** to get started.")
else:
    st.header(f"📌 {result['title']}")

    tab_summary, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary & Insights", "📝 Full Transcript", "💬 Chat"]
    )

    # ---- Summary tab -----------------------------------------------------
    with tab_summary:
        st.subheader("Summary")
        st.write(result["summary"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("✅ Action Items")
            st.write(result["action_items"])
        with col2:
            st.subheader("🔑 Key Decisions")
            st.write(result["key_decisions"])
        with col3:
            st.subheader("❓ Open Questions")
            st.write(result["open_questions"])

    # ---- Transcript tab ----------------------------------------------------
    with tab_transcript:
        st.subheader("Full Transcript")
        st.text_area("Transcript", result["transcript"], height=500, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=result["transcript"],
            file_name=f"{result['title']}_transcript.txt",
            mime="text/plain",
        )

    # ---- Chat tab ----------------------------------------------------------
    with tab_chat:
        st.subheader("Chat with your meeting")

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("Ask a question about this video…")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Error answering question: {e}"
                st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))