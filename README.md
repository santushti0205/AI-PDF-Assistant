# AI PDF Assistant

A RAG-based PDF question-answering application built with Python, Streamlit, SentenceTransformers, and Gemini.

## Features

- Upload a PDF
- Extract text using pypdf
- Split text into overlapping chunks
- Generate embeddings using SentenceTransformers
- Retrieve relevant chunks using cosine similarity
- Generate grounded answers using Gemini
- Interactive Streamlit web interface

## Tech Stack

- Python
- Streamlit
- pypdf
- SentenceTransformers
- scikit-learn
- Gemini API

## How It Works

1. User uploads a PDF.
2. Text is extracted from the document.
3. The text is split into overlapping chunks.
4. Each chunk is converted into an embedding.
5. The user's question is also converted into an embedding.
6. Cosine similarity finds the most relevant chunks.
7. Relevant chunks are passed to Gemini as context.
8. Gemini generates the final answer.

## Installation

Install dependencies:

```bash
py -m pip install -r requirements.txt