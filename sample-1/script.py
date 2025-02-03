from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pdfplumber
from transformers import pipeline

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Chunk text into manageable pieces
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Create embeddings using SentenceTransformer
def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# Find the most relevant chunk for a query
def find_relevant_chunk(query, chunks, embeddings, model):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index]

# Answer the query using Hugging Face's QA pipeline
def answer_query(context, query):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

# Main function to process PDF and answer questions
def process_pdf_and_answer(pdf_path):
    # Extract and chunk the PDF text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Create embeddings
    embeddings, model = create_embeddings(chunks)
    return (chunks, embeddings, model)

# Find the most relevant chunk
def final(query, chunks, embeddings, model):
    relevant_chunk = find_relevant_chunk(query, chunks, embeddings, model)

    # Answer the query using the most relevant chunk
    answer = answer_query(relevant_chunk, query)
    return answer

# Example usage
pdf_path = ''  # Replace with your PDF file path
queries = ["What is the main topic of the document?", "Explain the link layer", "Who is the editor of this"]
chunks, embeddings, model = process_pdf_and_answer(pdf_path)
answers = []

# Iterate over the queries and call process_pdf_and_answer for each
for query in queries:
    answer = final(query, chunks, embeddings, model)
    answers.append(answer)

print("Answers:", answers)
