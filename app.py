import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from docx import Document
from docx.shared import Pt
import tempfile
import os
import re

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Audio / YouTube → Transcript DOCX",
    layout="wide"
)

st.title("🎙️ Audio / YouTube → Transcript & Notes (DOCX)")
st.caption("Gemini API powered | Streamlit Community Ready")

# ----------------------------
# Gemini Setup (from secrets)
# ----------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    st.error("❌ Gemini API key missing. Please add it in secrets.toml")
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# Helpers
# ----------------------------
def extract_youtube_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{6,})",
        r"youtu\.be/([a-zA-Z0-9_-]{6,})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{6,})",
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(url):
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    lines = []
    for t in transcript:
        start = round(t["start"], 2)
        text = t["text"].replace("\n", " ")
        lines.append(f"[{start}s] {text}")
    return "\n".join(lines)


def generate_structured_notes(raw_text, language, style):
    prompt = f"""
You are a professional transcription editor.

TASK:
Convert the given transcript into a clean, structured Markdown document.

RULES:
- Language: {language}
- Style: {style}
- Use headings and subheadings
- Improve readability
- Do NOT change meaning

STRUCTURE:
# Title
## Key Takeaways
## Detailed Transcript
## Summary

TRANSCRIPT:
{raw_text}
"""

    response = model.generate_content(prompt)
    return response.text


def create_docx(title, content):
    doc = Document()
    doc.add_heading(title, level=0)

    for line in content.splitlines():
        line = line.strip()
        if not line:
            doc.add_paragraph("")
            continue

        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("- "):
            p = doc.add_paragraph(line[2:], style="List Bullet")
            for r in p.runs:
                r.font.size = Pt(11)
        else:
            p = doc.add_paragraph(line)
            for r in p.runs:
                r.font.size = Pt(11)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        path = tmp.name

    with open(path, "rb") as f:
        data = f.read()

    os.remove(path)
    return data

# ----------------------------
# UI
# ----------------------------
tab1, tab2 = st.tabs(["▶️ YouTube Link", "📝 Paste Transcript"])

raw_text = None

with tab1:
    st.subheader("YouTube Video")
    yt_url = st.text_input("Paste YouTube URL")

    if yt_url:
        try:
            raw_text = get_youtube_transcript(yt_url)
            st.success("Transcript fetched successfully")
            with st.expander("Preview Transcript"):
                st.text(raw_text[:3000])
        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Paste Transcript")
    pasted_text = st.text_area(
        "Paste raw transcript here",
        height=250,
        placeholder="Paste audio / video transcript here..."
    )

    if pasted_text.strip():
        raw_text = pasted_text.strip()

# ----------------------------
# Settings
# ----------------------------
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    language = st.selectbox("Output Language", ["English", "Hindi", "Hinglish"])
with col2:
    style = st.selectbox(
        "Document Style",
        ["Professional Notes", "Lecture Notes", "Meeting Minutes", "Podcast Notes"]
    )
with col3:
    file_name = st.text_input("DOCX File Name", "transcript_notes.docx")

title = st.text_input("Document Title", "Transcript & Notes")

# ----------------------------
# Generate
# ----------------------------
st.divider()
generate = st.button("🚀 Generate DOCX", use_container_width=True)

if generate:
    if not raw_text or len(raw_text) < 50:
        st.error("Please provide a transcript or YouTube link.")
        st.stop()

    with st.spinner("Generating structured transcript using Gemini..."):
        structured = generate_structured_notes(raw_text, language, style)

    st.success("Done!")

    st.markdown("### 📄 Preview")
    st.markdown(structured)

    docx = create_docx(title, structured)

    st.download_button(
        "⬇️ Download DOCX",
        data=docx,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True
    )
