import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load the processed dataset
try:
    qa_data = pd.read_csv('./qa_data_processed.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: qa_data_processed.csv not found.")
    exit(1)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print("Embedding model initialized successfully!")

# Prepare documents and metadata
if 'questions_title' not in qa_data.columns or 'answers_body' not in qa_data.columns:
    print("Error: Required columns 'questions_title' or 'answers_body' not found.")
    print("Columns:", qa_data.columns.tolist())
    exit(1)

questions = qa_data['questions_title'].tolist()
answers = qa_data['answers_body'].tolist()
metadatas = [{'answers_body': answer} for answer in answers]

# Create Chroma vector store
try:
    vectorstore = Chroma.from_texts(
        texts=questions,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name='career_advice',
        persist_directory='./chroma_db'
    )
    print("Chroma vector store created successfully!")
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit(1)