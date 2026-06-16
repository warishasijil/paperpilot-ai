import os
import io
import requests
import streamlit as st
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL") or st.secrets.get("N8N_WEBHOOK_URL")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


st.set_page_config(
    page_title="PaperPilot AI",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# BASIC UI STYLING
# -----------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero-box {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #475569;
    }

    .feature-card {
        padding: 18px;
        border-radius: 14px;
        background-color: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# FUNCTIONS
# -----------------------------

def extract_text_from_pdf(uploaded_pdf):
    pdf_bytes = uploaded_pdf.read()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        full_text += f"\n\n--- PDF Page {page_number} ---\n"
        full_text += text

    return full_text, len(document)


def load_uploaded_image(uploaded_image):
    image_bytes = uploaded_image.read()
    image = Image.open(io.BytesIO(image_bytes))
    return image


def analyze_image_notes_with_vision(image):
    if not GEMINI_API_KEY:
        return "Gemini API key not found. Please check your .env file."

    prompt = """
You are PaperPilot AI using Vision AI.

The user uploaded an image that may contain handwritten notes, printed notes, a textbook page, a table, a diagram, a whiteboard, or a screenshot.

Your task:
1. Extract all readable text from the image.
2. If it is handwritten, interpret it as accurately as possible.
3. Identify the topic.
4. Summarize the content.
5. Explain any diagram/table/chart if visible.
6. Convert the content into clean study notes.

Return your answer in this structure:

Extracted Text:
[write the extracted text here]

Topic:
[write topic here]

Summary:
[write summary here]

Important Points:
- point 1
- point 2
- point 3

Diagram/Table Explanation:
[explain if present, otherwise say "No clear diagram or table visible."]
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content([prompt, image])

    return response.text


def generate_content_summary(text, source_name="uploaded content"):
    if not GEMINI_API_KEY:
        return "Gemini API key not found. Please check your .env file."

    shortened_text = text[:30000]

    prompt = f"""
You are PaperPilot AI, an academic research assistant.

Read the following {source_name} and generate a clear structured summary.

Your response must include:

1. Title/topic if available
2. Main subject
3. Aim/purpose
4. Methodology or approach if available
5. Key findings or key points
6. Practical importance
7. Simple explanation in beginner-friendly language

Content:
{shortened_text}
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    return response.text


def split_text_into_chunks(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def create_tfidf_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks)
    return vectorizer, chunk_vectors


def retrieve_relevant_chunks(question, chunks, vectorizer, chunk_vectors, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    top_chunks = []
    for index in top_indices:
        top_chunks.append(chunks[index])

    return top_chunks


def answer_question_with_rag(question, source_text, source_label):
    if not source_text.strip():
        return f"No {source_label} content is available to answer from."

    chunks = split_text_into_chunks(source_text)
    vectorizer, chunk_vectors = create_tfidf_index(chunks)

    relevant_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        vectorizer,
        chunk_vectors,
        top_k=3
    )

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are PaperPilot AI, an academic research assistant.

The user wants an answer based on: {source_label}

Answer the user's question using ONLY the provided context.

Rules:
- Do not make up information.
- If the context does not contain the answer, say:
  "The uploaded content does not provide enough information to answer this."
- Keep the answer clear and beginner-friendly.
- Mention whether the answer came from PDF content or image notes.

Context:
{context}

User question:
{question}
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    return response.text


def convert_text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


def send_to_n8n(source, question, answer, summary):
    if not N8N_WEBHOOK_URL:
        return False, "N8N_WEBHOOK_URL not found. Please add it to your .env file."

    payload = {
        "source": source,
        "question": question,
        "answer": answer,
        "summary": summary
    }

    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=20)

        if response.status_code in [200, 201]:
            return True, "Report successfully sent to n8n automation."
        else:
            return False, f"n8n returned status code {response.status_code}: {response.text}"

    except Exception as e:
        return False, f"Error sending data to n8n: {e}"


# -----------------------------
# HERO SECTION
# -----------------------------

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">📄 PaperPilot AI</div>
        <div class="hero-subtitle">
            An AI research assistant that reads papers, understands image notes,
            answers questions with RAG, supports speech interaction, and sends reports through automation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown("<div class='feature-card'><b>📄 PDF AI</b><br>Reads research papers</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("<div class='feature-card'><b>👁️ Vision AI</b><br>Understands image notes</div>", unsafe_allow_html=True)

with col_c:
    st.markdown("<div class='feature-card'><b>🎙️ Speech AI</b><br>Voice input and audio output</div>", unsafe_allow_html=True)

with col_d:
    st.markdown("<div class='feature-card'><b>⚙️ Automation</b><br>Sends reports using n8n</div>", unsafe_allow_html=True)

st.divider()


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("PaperPilot Menu")
    st.write("Use this app to:")
    st.write("✅ Upload research papers")
    st.write("✅ Upload handwritten/printed notes")
    st.write("✅ Ask research questions")
    st.write("✅ Generate summaries")
    st.write("✅ Send reports via n8n")

    st.divider()

    st.caption("Recommended demo flow:")
    st.caption("1. Upload PDF")
    st.caption("2. Generate summary")
    st.caption("3. Ask methodology question")
    st.caption("4. Convert answer to speech")
    st.caption("5. Send report to n8n")


# -----------------------------
# UPLOAD SECTION
# -----------------------------

st.header("1. Upload Your Research Material")

col1, col2 = st.columns(2)

with col1:
    uploaded_pdf = st.file_uploader(
        "Upload research paper PDF",
        type=["pdf"]
    )

with col2:
    uploaded_image = st.file_uploader(
        "Upload image of notes / writing / textbook page",
        type=["jpg", "jpeg", "png"]
    )


if uploaded_pdf is not None:
    st.success("PDF uploaded successfully!")

    st.write("PDF file name:", uploaded_pdf.name)
    st.write("PDF file size:", round(uploaded_pdf.size / 1024, 2), "KB")

    with st.spinner("Extracting text from PDF..."):
        pdf_text, total_pages = extract_text_from_pdf(uploaded_pdf)

    st.session_state["pdf_text"] = pdf_text

    st.success(f"PDF text extracted from {total_pages} pages.")
    st.write("PDF extracted characters:", len(pdf_text))

    with st.expander("View extracted PDF text"):
        st.text_area(
            "PDF text",
            pdf_text,
            height=250
        )


if uploaded_image is not None:
    st.success("Image uploaded successfully!")

    image = load_uploaded_image(uploaded_image)

    st.image(
        image,
        caption="Uploaded image for Vision AI analysis",
        width=500
    )

    if st.button("Analyze Image with Vision AI"):
        with st.spinner("Gemini Vision is reading the image..."):
            image_analysis = analyze_image_notes_with_vision(image)

        st.session_state["image_text"] = image_analysis

    if "image_text" in st.session_state:
        st.subheader("Vision AI Extracted Notes")
        st.markdown(st.session_state["image_text"])

        with st.expander("View image text used for RAG"):
            st.text_area(
                "Image-derived text",
                st.session_state["image_text"],
                height=250
            )


st.divider()


# -----------------------------
# SUMMARY SECTION
# -----------------------------

st.header("2. Generate Summary")

summary_source = st.radio(
    "Choose what you want to summarize:",
    ["PDF only", "Image only"],
    horizontal=True
)

if st.button("Generate Summary", use_container_width=True):
    pdf_text = st.session_state.get("pdf_text", "")
    image_text = st.session_state.get("image_text", "")

    if summary_source == "PDF only":
        selected_text = pdf_text
        source_name = "PDF paper"
    else:
        selected_text = image_text
        source_name = "image notes"

    if not selected_text.strip():
        st.warning("No content available for the selected source.")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_content_summary(selected_text, source_name)

        st.session_state["latest_summary"] = summary
        st.session_state["latest_summary_source"] = source_name
        st.markdown(summary)


if "latest_summary" in st.session_state:
    with st.expander("View latest generated summary"):
        st.markdown(st.session_state["latest_summary"])


st.divider()


# -----------------------------
# QUESTION SECTION
# -----------------------------

st.header("3. Ask Questions Using Text or Voice")

question_source = st.radio(
    "Choose the source for answering:",
    ["PDF only", "Image only"],
    horizontal=True
)

st.subheader("Quick Research Questions")

q1, q2, q3, q4 = st.columns(4)

with q1:
    if st.button("🎯 Aim", use_container_width=True):
        st.session_state["current_question"] = "What is the main aim of this study?"

with q2:
    if st.button("🧪 Methodology", use_container_width=True):
        st.session_state["current_question"] = "What methodology was used in this research paper?"

with q3:
    if st.button("📊 Findings", use_container_width=True):
        st.session_state["current_question"] = "What are the main findings of this paper?"

with q4:
    if st.button("⚠️ Limitations", use_container_width=True):
        st.session_state["current_question"] = "What limitations are mentioned in this paper?"

q5, q6, q7, q8 = st.columns(4)

with q5:
    if st.button("💡 Contribution", use_container_width=True):
        st.session_state["current_question"] = "What is the main contribution or novelty of this paper?"

with q6:
    if st.button("🔮 Future Work", use_container_width=True):
        st.session_state["current_question"] = "What future work does this paper suggest?"

with q7:
    if st.button("🧠 Simple Explanation", use_container_width=True):
        st.session_state["current_question"] = "Explain this content in simple beginner-friendly language."

with q8:
    if st.button("📝 Exam Notes", use_container_width=True):
        st.session_state["current_question"] = "Convert this content into concise study notes."

st.subheader("Voice Question")

voice_question = speech_to_text(
    language="en",
    use_container_width=True,
    just_once=True,
    key="voice_question"
)

if voice_question:
    st.session_state["current_question"] = voice_question
    st.success(f"Voice detected: {voice_question}")

st.subheader("Typed Question")

user_question = st.text_input(
    "Ask a question",
    value=st.session_state.get("current_question", ""),
    placeholder="Example: What is the methodology of this study?"
)

if user_question:
    st.session_state["current_question"] = user_question


if st.button("Ask PaperPilot", use_container_width=True):
    final_question = st.session_state.get("current_question", "")

    if final_question.strip() == "":
        st.warning("Please type or speak a question first.")
    else:
        pdf_text = st.session_state.get("pdf_text", "")
        image_text = st.session_state.get("image_text", "")

        if question_source == "PDF only":
            selected_text = pdf_text
            source_label = "PDF content only"
        else:
            selected_text = image_text
            source_label = "image notes only"

        if not selected_text.strip():
            st.warning("No content available for the selected source.")
        else:
            with st.spinner("Retrieving relevant content and generating answer..."):
                rag_answer = answer_question_with_rag(
                    final_question,
                    selected_text,
                    source_label
                )

            st.session_state["last_question"] = final_question
            st.session_state["last_answer"] = rag_answer
            st.session_state["last_source"] = source_label

if "last_answer" in st.session_state:
    st.subheader("Answer")
    st.markdown(st.session_state["last_answer"])

    audio_col, download_col = st.columns(2)

    with audio_col:
        if st.button("🔊 Convert Answer to Speech", use_container_width=True):
            with st.spinner("Creating audio answer..."):
                audio = convert_text_to_speech(st.session_state["last_answer"])

            st.audio(audio, format="audio/mp3")

    with download_col:
        st.download_button(
            label="⬇️ Download Answer",
            data=st.session_state["last_answer"],
            file_name="paperpilot_answer.txt",
            mime="text/plain",
            use_container_width=True
        )


st.divider()


# -----------------------------
# AUTOMATION SECTION
# -----------------------------

st.header("4. Automation: Send Report")

st.write(
    "Send your latest question, answer, and summary to n8n. "
    "n8n can then email the report or save it to another app."
)

if st.button("Send Report to n8n", use_container_width=True):
    latest_source = st.session_state.get(
        "last_source",
        st.session_state.get("latest_summary_source", "No source selected")
    )
    latest_question = st.session_state.get("last_question", "No question asked yet.")
    latest_answer = st.session_state.get("last_answer", "No answer generated yet.")
    latest_summary = st.session_state.get("latest_summary", "No summary generated yet.")

    success, message = send_to_n8n(
        source=latest_source,
        question=latest_question,
        answer=latest_answer,
        summary=latest_summary
    )

    if success:
        st.success(message)
    else:
        st.error(message)