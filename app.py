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
st.caption("Upload Audio/Video or use YouTube link, then generate a DOCX with headings/subheadings.")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    st.error("❌ Gemini API key missing. Add it in `.streamlit/secrets.toml` as GEMINI_API_KEY.")
    st.stop()

# You can switch model if you want:
# gemini-1.5-flash (fast) or gemini-1.5-pro (better quality)
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
    Works for many audio/video mime types (depends on Gemini support).
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        uploaded = genai.upload_file(tmp_path, mime_type=mime_type)

        prompt = """
Transcribe this media accurately.
Return transcript as plain text.
If you can infer timestamps, include them in [00:00] format, otherwise skip timestamps.
"""

        resp = model.generate_content([prompt, uploaded])
        return resp.text.strip()
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
    return resp.text


def create_docx(title: str, content_md: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=0)

    for line
