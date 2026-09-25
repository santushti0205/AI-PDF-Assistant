from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
reader = PdfReader("notes.pdf")
text = ""
for i, page in enumerate(reader.pages[:35]):
    print("Reading page:", i + 1)

    page_text = page.extract_text()

    if page_text:
        text += page_text

print("Finished reading PDF")
print(text[:500]) #take the characters from the beginning of text up to character index 499


# 2. Split text into chunks
chunk_size = 1000
overlap = 200

chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print("\nCHUNK", i)
    print(chunk[:300])
    print("-" * 50)


# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Convert PDF chunks into embeddings
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)


# 5. User question
client = genai.Client()
while True:
    question = input("\nAsk a question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Chatbot stopped.")
        break

    question_embedding = model.encode([question])

    similarities = cosine_similarity(question_embedding, embeddings)

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

    print("\nAnswer:")
    print(answer)


# 6. Convert question into embedding
question_embedding = model.encode([question])


# 7. Compare question with all PDF chunks
similarities = cosine_similarity(question_embedding, embeddings)

print(similarities)
best_chunk_index=similarities.argmax()
best_chunk=chunks[best_chunk_index]
print("best matching chunk:")
print(best_chunk)
print("similarity score:", similarities[0][best_chunk_index])
top_indices=similarities[0].argsort()[-5:][::-1]
for index in top_indices:
    print("\n similarity score:",similarities[0][index])
    print(chunks[index])
    print("-"*50)
context=""
for index in top_indices:
    context += chunks[index] + "\n"
print("combined context:")
print(context)
prompt=f"""
Answer the question using only the context below.
context:
{context}
question:
{question}
answer:
"""
print(prompt)
import os
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

answer = response.text

print("\nFinal Answer:")
print(answer)


