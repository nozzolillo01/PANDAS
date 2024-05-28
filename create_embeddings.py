"""
Questo script legge il contenuto di tutti i file .txt in una directory specificata,
divide i documenti in chunk (con RecursiveCharacterTextSplitter) e ottiene le embeddings
per ogni chunk utilizzando il modello di embedding di PremAI. Le embeddings e i chunk
vengono salvati in un file pickle.
"""

# Link alla documentazione di PremAI:
# https://docs.premai.io/introduction

import os  # Importa la libreria os per le operazioni di sistema
import glob  # Importa la libreria glob per trovare i percorsi dei file che corrispondono a un pattern specifico
import pickle  # Importa la libreria pickle per la serializzazione degli oggetti
from langchain_community.embeddings import PremAIEmbeddings  # Importa l'embedder di PremAI di LangChain
from config import PREMAI_API_KEY  # Importa la chiave API dal file di configurazione
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Importa il text splitter di LangChain

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY

# Definisci il modello e inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4316, model=model)  # Crea un oggetto embedder con il modello specificato

# Directory contenente i file .txt
directory_path = r"estrai_doc\txt"

# Ottieni una lista di tutti i file .txt nella directory
txt_files = glob.glob(os.path.join(directory_path, '*.txt'))

# Leggi il contenuto di ogni file e conservalo in una lista
documents = []
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()  # Legge il contenuto del file
        documents.append(document_content)  # Aggiunge il contenuto del file alla lista dei documenti

# Definisci il text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Dimensione dei chunk
    chunk_overlap=100,  # Sovrapposizione tra i chunk
    length_function=len,  # Funzione per calcolare la lunghezza del testo
    is_separator_regex=False,  # Indica se i separatori sono espressi come regex
    separators=[
        "\n\n",  # Doppio newline
        "\n",  # Singolo newline
        " ",  # Spazio
        ".",  # Punto
        ",",  # Virgola
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",  # Separatore vuoto
    ],
)

# Dividi i documenti in chunk
all_chunks = []
for document in documents:
    chunks = text_splitter.split_text(document)  # Divide il documento in chunk
    all_chunks.extend(chunks)  # Aggiunge i chunk alla lista di tutti i chunk

# Ottieni le embeddings per ogni chunk
chunk_embeddings = embedder.embed_documents(all_chunks)  # Genera le embeddings per ogni chunk

# Salva i chunk e le embeddings in un file pickle
with open('chunk_embeddings.pkl', 'wb') as f:
    pickle.dump((all_chunks, chunk_embeddings), f)  # Serializza e salva i chunk e le loro embeddings

print("Chunk embeddings created and saved successfully.")  # Messaggio di conferma
