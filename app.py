import os
import re
import time
import tempfile

import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

from docx import Document
from docx.shared import Pt


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Aiclex Transcript DOCX", layout="wide")


# ----------------------------
# First-time Loading Screen
# ----------------------------
if "booted" not in st.session_state:
    st.session_state.booted = True

    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;">
            <h1 style="margin-bottom: 10px;">Aiclex Technologies</h1>
            <h3 style="margin-top: 0px; opacity: 0.8;">Made by Umesh Kumar</h3>
            <p style="margin-top: 30px; font-size: 18px;">Loading…</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.spinner("Initializing..."):
        time.sleep(2)
    st.rerun()


# ----------------------------
# Gemini setup (from secrets)
# ----------------------------
st.title("🎙️ Aiclex Transcript → Structured Notes → DOCX")
st.caption("Select a source (Audio/Video upload or YouTube link), then generate a DOCX with headings/subheadings.")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    st.error("❌ Gemini API key missing. Add it in `.streamlit/secrets.toml` as GEMINI_API_KEY.")
    st.stop()

MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)


# ----------------------------
# Helpers
# ----------------------------
def extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"v=([a-zA-Z0-9_-]{6,})",
        r"youtu\.be/([a-zA-Z0-9_-]{6,})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{6,})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript_text(url: str) -> str:
    vid = extract_youtube_id(url)
    if not vid:
        raise ValueError("Invalid YouTube link (video id not found).")

    transcript = YouTubeTranscriptApi.get_transcript(vid)
    lines = []
    for t in transcript:
        start = round(float(t.get("start", 0.0)), 2)
        text = (t.get("text", "") or "").replace("\n", " ")
        lines.append(f"[{start}s] {text}")
    return "\n".join(lines)


def gemini_transcribe_file(file_bytes: bytes, mime_type: str) -> str:
    """
    Upload file to Gemini and ask it to transcribe.
    Works for many audio/video mime types (Gemini support may vary).
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        uploaded = genai.upload_file(tmp_path, mime_type=mime_type)

        prompt = (
            "Transcribe this media accurately.\n"
            "Return transcript as plain text.\n"
            "If you can infer timestamps, include them in [00:00] format, otherwise skip timestamps."
        )

        resp = model.generate_content([prompt, uploaded])
        return (resp.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def gemini_make_structured_doc(raw_transcript: str, language: str, style: str) -> str:
    prompt = f"""
You are a professional transcription editor and note writer.

Convert the given transcript into a clean, well-structured Markdown document.

Language: {language}
Style: {style}

Output format MUST be Markdown with headings/subheadings:
# Title
## Key Takeaways
## Detailed Transcript
## Summary

Rules:
- Improve readability (remove filler words lightly) but do NOT change meaning.
- Use bullet points where helpful.
- Keep it clean and usable as a document.
- If transcript seems incomplete, mention it politely in Summary.

TRANSCRIPT:
{raw_transcript}
"""
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def create_docx(title: str, content_md: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)

    for line in content_md.splitlines():
        s = line.strip()

        if not s:
            doc.add_paragraph("")
            continue

        if s.startswith("### "):
            doc.add_heading(s[4:], level=3)

        elif s.startswith("## "):
            doc.add_heading(s[3:], level=2)

        elif s.startswith("# "):
            doc.add_heading(s[2:], level=1)

        elif s.startswith("- ") or s.startswith("* "):
            p = doc.add_paragraph(s[2:], style="List Bullet")
            for run in p.runs:
                run.font.size = Pt(11)

        else:
            p = doc.add_paragraph(s)
            for run in p.runs:
                run.font.size = Pt(11)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        path = tmp.name

    with open(path, "rb") as f:
        data = f.read()

    try:
        os.remove(path)
    except Exception:
        pass

    return data


# ----------------------------
# UI: Source selector dropdown
# ----------------------------
st.divider()
source = st.selectbox(
    "Select Source",
    ["YouTube Link", "Upload Video (My PC)", "Upload Audio (My PC)"],
    index=0
)

raw_transcript = None

if source == "YouTube Link":
    yt_url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if yt_url:
        try:
            raw_transcript = get_youtube_transcript_text(yt_url)
            st.success("✅ YouTube transcript fetched.")
            with st.expander("Preview transcript"):
                st.text(raw_transcript[:4000] + ("..." if len(raw_transcript) > 4000 else ""))
        except Exception as e:
            st.error(f"❌ Transcript fetch failed: {e}")
            st.info("If video has no transcript, use upload option or paste transcript manually (below).")

elif source == "Upload Video (My PC)":
    up = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv", "webm"])
    if up:
        st.video(up)
        if st.button("🧾 Transcribe Video with Gemini", use_container_width=True):
            with st.spinner("Transcribing video..."):
                ext = os.path.splitext(up.name)[1].lower()
                mime = "video/mp4" if ext == ".mp4" else "video/*"
                raw_transcript = gemini_transcribe_file(up.getvalue(), mime)
            st.success("✅ Transcription ready.")
            with st.expander("Preview transcript"):
                st.text(raw_transcript[:4000] + ("..." if len(raw_transcript) > 4000 else ""))

elif source == "Upload Audio (My PC)":
    up = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "aac", "ogg"])
    if up:
        st.audio(up)
        if st.button("🧾 Transcribe Audio with Gemini", use_container_width=True):
            with st.spinner("Transcribing audio..."):
                ext = os.path.splitext(up.name)[1].lower()
                if ext == ".mp3":
                    mime = "audio/mpeg"
                elif ext == ".wav":
                    mime = "audio/wav"
                elif ext == ".m4a":
                    mime = "audio/mp4"
                else:
                    mime = "audio/*"
                raw_transcript = gemini_transcribe_file(up.getvalue(), mime)
            st.success("✅ Transcription ready.")
            with st.expander("Preview transcript"):
                st.text(raw_transcript[:4000] + ("..." if len(raw_transcript) > 4000 else ""))


# ----------------------------
# Optional manual transcript override (always available)
# ----------------------------
st.divider()
st.subheader("📝 Optional: Paste/Edit Transcript (Override)")
manual = st.text_area(
    "If you want, paste or edit transcript here (this will be used if not empty).",
    height=220,
    placeholder="Paste transcript here..."
)
if manual.strip():
    raw_transcript = manual.strip()
    st.info("Using pasted transcript.")


# ----------------------------
# Document settings
# ----------------------------
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    language = st.selectbox("Output Language", ["English", "Hindi", "Hinglish"], index=2)
with c2:
    style = st.selectbox(
        "Document Style",
        ["Professional Notes", "Lecture Notes", "Meeting Minutes", "Podcast Notes"],
        index=0
    )
with c3:
    file_name = st.text_input("DOCX File Name", "aiclex_transcript_notes.docx")

doc_title = st.text_input("Document Title", "Aiclex Transcript & Notes")


# ----------------------------
# Generate structured doc + DOCX
# ----------------------------
st.divider()
if st.button("🚀 Generate Structured DOCX", type="primary", use_container_width=True):
    if not raw_transcript or len(raw_transcript) < 40:
        st.error("Please provide transcript (YouTube / Transcribe audio/video / paste transcript).")
        st.stop()

    with st.spinner("Generating structured headings/subheadings using Gemini..."):
        structured_md = gemini_make_structured_doc(raw_transcript, language, style)

    st.success("✅ Structured document ready!")
    st.markdown("### Preview")
    st.markdown(structured_md)

    docx_bytes = create_docx(doc_title, structured_md)
    st.download_button(
        "⬇️ Download DOCX",
        data=docx_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
