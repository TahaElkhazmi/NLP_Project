import json
import os
import logging
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# Configure OpenAI API Key
def configure_openai_api_key():
    if os.getenv("OPENAI_API_KEY") is None:
        with open('key.txt', 'r') as f:
            os.environ["OPENAI_API_KEY"] = f.readline().strip()
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
    print("OpenAI API key configured")


configure_openai_api_key()


# Load data from multiple JSON files
def load_json_data(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for category, entries in data.items():
            for entry in entries:
                doc = Document(
                    page_content=entry["content"],
                    metadata={
                        "title": entry["lecture_title"],
                        "url": entry["lecture_url"],
                        "category": category,
                        "path": entry["path"]
                    }
                )
                documents.append(doc)
    return documents


# Split documents into chunks
def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


# Create vector store
def create_vector_store(documents, vector_store_path="./vector_store"):
    if not os.path.exists(vector_store_path):
        os.makedirs(vector_store_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vector_store_path,
    )
    vector_store.persist()
    return vector_store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    json_file_paths = ["data1.json", "data2.json", "data3.json", "data4.json"]
    vector_store_path = "./vector_store"

    logging.info("Loading JSON data from multiple files...")
    documents = load_json_data(json_file_paths)
    logging.info(f"Loaded {len(documents)} documents from JSON files.")

    logging.info("Splitting documents into chunks...")
    split_documents = chunk_documents(documents)
    logging.info(f"Total chunks created: {len(split_documents)}")

    logging.info("Creating vector store...")
    vector_store = create_vector_store(split_documents, vector_store_path)
    logging.info("Vector store successfully created and persisted.")
