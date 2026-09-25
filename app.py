import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
st.title("AI PDF ASSISTANCE")
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    st.success("PDF uploaded successfully!")

    chunk_size = 1000
    overlap = 200

    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    client = genai.Client()

    question = st.text_input("Ask a question about the document")

    if question:

        question_embedding = model.encode([question])

        similarities = cosine_similarity(
            question_embedding,
            embeddings
        )

        top_indices = similarities[0].argsort()[-5:][::-1]

        context = ""

        for index in top_indices:
            context += chunks[index] + "\n"

        prompt = f"""
Answer the question using only the context below.

If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        answer = response.text

        st.subheader("Answer")
        st.write(answer)


