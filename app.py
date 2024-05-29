import streamlit as st  # Importa la libreria Streamlit per creare l'interfaccia web
import os  # Importa la libreria os per le operazioni di sistema
import pickle  # Importa la libreria pickle per la serializzazione degli oggetti
from langchain_core.messages import HumanMessage, SystemMessage  # Importa le classi di messaggi di LangChain
from langchain_community.chat_models import ChatPremAI  # Importa il modello di chat PremAI di LangChain
from sklearn.metrics.pairwise import cosine_similarity  # Importa la funzione per calcolare la similarità coseno
import numpy as np  # Importa la libreria numpy per le operazioni numeriche
from langchain_community.embeddings import PremAIEmbeddings  # Importa l'embedder di PremAI di LangChain
from config import PREMAI_API_KEY  # Importa la chiave API dal file di configurazione

# Imposta la chiave API come variabile d'ambiente
os.environ["PREMAI_API_KEY"] = PREMAI_API_KEY

# Carica i chunk e le embeddings dal file pickle
with open('chunk_embeddings.pkl', 'rb') as f:
    all_chunks, chunk_embeddings = pickle.load(f)  # Deserializza i chunk di testo e le loro embeddings

# Inizializza l'embedder
model = "text-embedding-3-large"
embedder = PremAIEmbeddings(project_id=4494, model=model)  # Crea un oggetto embedder con il modello specificato

# Funzione per trovare i chunk più simili alla query
def find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks, top_k=10):
    similarities = cosine_similarity([query_embedding], chunk_embeddings).flatten()  # Calcola la similarità coseno tra la query e i chunk
    most_similar_indices = similarities.argsort()[-top_k:][::-1]  # Ordina gli indici dei chunk per similarità in ordine decrescente
    return [(all_chunks[i], similarities[i]) for i in most_similar_indices]  # Restituisce i top_k chunk più simili con i loro punteggi

# Streamlit app
st.set_page_config(page_title="Chatbot Pandas 🐼✨", page_icon="✨")  # Configura il titolo e l'icona della pagina Streamlit

# Definisce lo stile della pagina
st.markdown("""
<style>
    .stApp {
        background-color: ##1F1F1F;
        font-family: 'Arial', sans-serif;
    }
    .header-text {
        color: #4f8bcc;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subheader-text {
        color: #2e3b4e;
        font-size: 1.5rem;
    }
    .user-input {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .response-box {
        padding: 1.5rem;
        background-color: #4C5261;
        color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Aggiunge il titolo e il sottotitolo dell'app
st.markdown('<div class="header-text">Benvenuto nel Chatbot Pandas 🐼✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Fai una domanda e il chatbot ti risponderà!</div>', unsafe_allow_html=True)

# Inizializza la lista dei messaggi nella sessione di Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []  # Se non esiste, crea una lista vuota per i messaggi

# Input di testo per la query dell'utente
user_query = st.text_input("Inserisci la tua domanda:", key="user_input", help="Scrivi qui la tua domanda", label_visibility="collapsed")

if user_query:  # Se l'utente ha inserito una query
    with st.spinner(''):  # Mostra uno spinner di caricamento
        query_embedding = embedder.embed_query(user_query)  # Crea l'embedding della query dell'utente

        most_similar_chunks = find_most_similar_chunks(query_embedding, chunk_embeddings, all_chunks)  # Trova i chunk più simili

        combined_text = " ".join([chunk for chunk, _ in most_similar_chunks])  # Combina i chunk più simili in un testo unico

        chat = ChatPremAI(project_id=4494)  # Inizializza il modello di chat
        system_message = SystemMessage(content=combined_text)  # Crea un messaggio di sistema con il testo combinato
        human_message = HumanMessage(content=user_query)  # Crea un messaggio umano con la query dell'utente
        response = chat.invoke([system_message, human_message])  # Genera una risposta usando il modello di chat

        st.session_state.messages.append({"role": "user", "content": user_query})  # Aggiunge la query dell'utente ai messaggi
        st.session_state.messages.append({"role": "bot", "content": response.content})  # Aggiunge la risposta del bot ai messaggi

# Visualizza i messaggi come una chat in ordine inverso
for message in reversed(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f'<div class="response-box" style="background-color: #1E1E1E; color: #FFFFFF;">{message["content"]}</div>', unsafe_allow_html=True)  # Messaggio dell'utente
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)  # Spazio tra i messaggi
    else:
        st.markdown(f'<div class="response-box">{message["content"]}</div>', unsafe_allow_html=True)  # Messaggio del bot
