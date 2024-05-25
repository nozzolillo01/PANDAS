from docx import Document
import re
import os

# Crea la cartella 'txt' se non esiste
if not os.path.exists('txt'):
    os.makedirs('txt')

# Percorso della cartella con i file .docx
docx_folder = "doc"

# Itera su ogni file nella cartella doc
for filename in os.listdir(docx_folder):
    if filename.endswith(".docx"):
        docx_file = os.path.join(docx_folder, filename)
        
        # Apri il file .docx
        doc = Document(docx_file)

        # Estrai testo da ogni paragrafo
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"

        # Salva il testo estratto in un file txt nella cartella txt
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join("txt", txt_filename)
        with open(txt_filepath, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        # Carica il testo estratto
        with open(txt_filepath, "r", encoding="utf-8") as file:
            text = file.read()

        # Suddividi il testo in base ai nomi (ogni entry inizia con un nome in maiuscolo seguito da dettagli)
        entries = re.split(r'\n(?=[A-Z\s]+\n)', text)

        # Pulisci e struttura ogni entry
        structured_data = []
        for entry in entries:
            lines = entry.strip().split('\n')
            if len(lines) > 1:
                structured_data.append({
                    "name": lines[0].strip(),
                    "details": " ".join(line.strip() for line in lines[1:] if line.strip())
                })

        # Verifica i dati strutturati
        #for data in structured_data[:5]:  # Visualizza i primi 5 elementi
            #print(data)
