import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

DB_PATH = "vectorstore"


def process_pdf(file_path):

    print("Loading PDF...")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    batch_size = 1000

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        print(f"Embedding batch {i} → {i + len(batch)}")

        vectordb.add_documents(batch)

    vectordb.persist()

    print("Vector database created successfully")


def reset_database():

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    os.makedirs(DB_PATH, exist_ok=True)

    print("Vector database reset")
