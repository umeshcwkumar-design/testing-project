import os
import re
import time
import mimetypes
import tempfile

import streamlit as st
import google.generativeai as genai
from google.generativeai import types
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
            <h1>Aiclex Technologies</h1>
            <h3>Made by Umesh Kumar</h3>
            <p style="font-size:18px;">Loading…</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(2)
    st.rerun()


# ----------------------------
# Gemini setup
# ----------------------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Gemini API key missing in secrets.toml")
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

st.title("🎙️ Aiclex Audio / Video / YouTube → DOCX")
st.caption("Upload media or paste YouTube link → transcript → structured notes → DOCX")


# ----------------------------
# Helpers
# ----------------------------
def extract_youtube_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{6,})",
        r"youtu\.be/([a-zA-Z0-9_-]{6,})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript(url):
    vid = extract_youtube_id(url)
    if not vid:
        raise ValueError("Invalid YouTube link")

    transcript = YouTubeTranscriptApi.get_transcript(vid)
    return "\n".join(
        [f"[{round(t['start'],2)}s] {t['text'].replace(chr(10),' ')}" for t in transcript]
    )


def wait_until_active(file_obj, timeout=180):
    start = time.time()
    while True:
        f = genai.get_file(file_obj.name)
        state = str(f.state).upper()
        if "ACTIVE" in state:
            return f
        if "FAILED" in state:
            raise RuntimeError("Gemini file processing failed")
        if time.time() - start > timeout:
            raise TimeoutError("File processing timeout")
        time.sleep(2)


def gemini_transcribe(file_bytes, filename):
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        uploaded = genai.upload_file(path, mime_type=mime)
        uploaded = wait_until_active(uploaded)

        media_part = types.Part.from_uri(uploaded.uri, mime_type=mime)

        prompt = (
            "Transcribe this media accurately.\n"
            "Return plain text transcript.\n"
            "Add timestamps if possible.\n"
            "Do not add explanations."
        )

        resp = model.generate_content([prompt, media_part])
        return resp.text.strip()

    finally:
        os.remove(path)


def make_structured(transcript, language, style):
    prompt = f"""
Convert this transcript into structured Markdown notes.

Language: {language}
Style: {style}

Format:
# Title
## Key Takeaways
## Detailed Transcript
## Summary

Transcript:
{transcript}
"""
    return model.generate_content(prompt).text.strip()


def create_docx(title, md):
    doc = Document()
    doc.add_heading(title, 0)

    for line in md.splitlines():
        s = line.strip()
        if not s:
            doc.add_paragraph("")
            continue
        if s.startswith("### "):
            doc.add_heading(s[4:], 3)
        elif s.startswith("## "):
            doc.add_heading(s[3:], 2)
        elif s.startswith("# "):
            doc.add_heading(s[2:], 1)
        elif s.startswith("- "):
            p = doc.add_paragraph(s[2:], style="List Bullet")
            for r in p.runs:
                r.font.size = Pt(11)
        else:
            p = doc.add_paragraph(s)
            for r in p.runs:
                r.font.size = Pt(11)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        data = open(tmp.name, "rb").read()
        os.remove(tmp.name)
        return data


# ----------------------------
# UI
# ----------------------------
source = st.selectbox(
    "Select Input Source",
    ["YouTube Link", "Upload Video (My PC)", "Upload Audio (My PC)"]
)

raw_text = None

if source == "YouTube Link":
    url = st.text_input("Paste YouTube URL")
    if url:
        raw_text = get_youtube_transcript(url)
        st.success("Transcript fetched")

elif source == "Upload Video (My PC)":
    file = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv"])
    if file and st.button("Transcribe Video"):
        raw_text = gemini_transcribe(file.getvalue(), file.name)
        st.success("Video transcribed")

elif source == "Upload Audio (My PC)":
    file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
    if file and st.button("Transcribe Audio"):
        raw_text = gemini_transcribe(file.getvalue(), file.name)
        st.success("Audio transcribed")


raw_text = st.text_area("Edit / Paste Transcript (optional)", raw_text or "", height=200)

language = st.selectbox("Language", ["English", "Hindi", "Hinglish"])
style = st.selectbox("Document Style", ["Professional", "Lecture", "Meeting", "Podcast"])
title = st.text_input("Document Title", "Aiclex Transcript & Notes")
filename = st.text_input("DOCX File Name", "aiclex_notes.docx")


if st.button("🚀 Generate DOCX", use_container_width=True):
    if len(raw_text.strip()) < 40:
        st.error("Transcript too short")
    else:
        with st.spinner("Generating structured document..."):
            md = make_structured(raw_text, language, style)
        st.markdown(md)
        docx = create_docx(title, md)
        st.download_button(
            "⬇️ Download DOCX",
            docx,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
