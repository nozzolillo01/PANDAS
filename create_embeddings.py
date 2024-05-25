"""
Questo script legge il contenuto di tutti i file .txt in una directory specificata, divide i documenti in chunk (con RecursiveCharacterTextSplitter) e ottiene le embeddings per ogni chunk utilizzando il modello di embedding di PremAI. Le embeddings e i chunk vengono salvati in un file pickle.
"""
#link documentazione di PremAI:
#https://docs.premai.io/introduction

import os
import glob
import pickle
from langchain_community.embeddings import PremAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Carica le variabili d'ambiente da un file .env
load_dotenv()

# Ottieni la chiave API da un file .env
PREMAI_API_KEY = os.getenv("PREMAI_API_KEY")

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY
# Definisci il modello e inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4316, model=model)

# Directory contenente i file .txt
directory_path = r"documents\txt"

# Ottieni una lista di tutti i file .txt nella directory
txt_files = glob.glob(os.path.join(directory_path, '*.txt'))

# Leggi il contenuto di ogni file e conservalo in una lista
documents = []
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()
        documents.append(document_content)

# magari si può provare character_text_splitter
# Definisci il text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
)

# Dividi i documenti in chunk
all_chunks = []
for document in documents:
    chunks = text_splitter.split_text(document)
    all_chunks.extend(chunks)

# Ottieni le embeddings per ogni chunk
chunk_embeddings = embedder.embed_documents(all_chunks)

# Salva i chunk e le embeddings in un file pickle
with open('chunk_embeddings.pkl', 'wb') as f:
    pickle.dump((all_chunks, chunk_embeddings), f)

print("Chunk embeddings created and saved successfully.")
