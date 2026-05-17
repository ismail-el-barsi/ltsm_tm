import streamlit as st
import numpy as np
import keras
from keras.datasets import imdb
import csv
import datetime
import os

try:
    from keras.utils import pad_sequences
except ImportError:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration & Constantes
VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = 'imdb_sentiment.keras'
LOG_FILE = 'inference_log.csv'

st.set_page_config(page_title="IMDB Analytics & Logs", page_icon="📈", layout="centered")

# =====================================================================
# SECURE LOGGING FUNCTION (Médiation des attaques de format)
# =====================================================================
def log_inference(input_text, prediction_label, confidence_score, log_file=LOG_FILE):
    """
    Loggue chaque inférence dans un CSV local de manière sécurisée.
    Gère nativement l'échappement des virgules et des sauts de ligne.
    """
    # Nettoyage cosmétique pour le snapshot du log (tronqué à 100 caractères)
    truncated_text = input_text[:100].replace("\n", " ") + "..." if len(input_text) > 100 else input_text
    timestamp = datetime.datetime.now().isoformat()
    
    # Vérification de l'existence du fichier pour écrire l'entête si nécessaire (Edge Case)
    file_exists = os.path.exists(log_file)
    
    # Mode 'a' (append) : crée le fichier s'il n'existe pas, sinon ajoute à la fin
    with open(log_file, mode='a', encoding='utf-8', newline='') as f:
        # quoting=csv.QUOTE_MINIMAL force le writer à envelopper le texte entre guillemets ("...") 
        # si la chaîne contient des virgules, des guillemets ou des retours à la ligne.
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        if not file_exists:
            writer.writerow(["Timestamp", "Input_Text_Snapshot", "Prediction", "Confidence"])
            
        writer.writerow([timestamp, truncated_text, prediction_label, f"{confidence_score:.2f}%"])

# =====================================================================
# CHARGEMENT DES RESSOURCES EN CACHE
# =====================================================================
@st.cache_resource
def load_resources():
    model = keras.models.load_model(MODEL_PATH)
    word_index = imdb.get_word_index()
    adjusted_index = {k: (v + 3) for k, v in word_index.items()}
    adjusted_index["<PAD>"] = 0
    adjusted_index["<START>"] = 1
    adjusted_index["<UNK>"] = 2
    return model, adjusted_index

model, imdb_index = load_resources()

# Preprocess Pipeline
def preprocess_text(text, word_index, max_len, vocab_size):
    clean_text = text.lower().replace(",", "").replace(".", "")
    words = clean_text.split()
    tokens = [1]
    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            tokens.append(word_index[word])
        else:
            tokens.append(2)
    return pad_sequences([tokens], maxlen=max_len, padding='pre', truncating='post')

# =====================================================================
# INTERFACE GRAPHIQUE
# =====================================================================
st.title("🎬 Analyseur de Sentiment IMDB + Monitoring")

user_input = st.text_area("Saisissez votre critique :", height=120)
st.caption("Note : les textes de plus de 200 mots sont tronqués.")

if st.button("Analyser", type="primary"):
    if not user_input.strip():
        st.warning("Veuillez entrer un texte valide.")
    else:
        processed = preprocess_text(user_input, imdb_index, MAX_LEN, VOCAB_SIZE)
        raw_pred = model.predict(processed, verbose=0)[0][0]
        
        # Détermination des métriques finales
        if raw_pred >= 0.5:
            label, confidence = "Positif", raw_pred * 100
            st.success(f"**Sentiment : {label}** ({confidence:.2f}%)")
        else:
            label, confidence = "Négatif", (1 - raw_pred) * 100
            st.error(f"**Sentiment : {label}** ({confidence:.2f}%)")
            
        # APPEL DU LOGGER : Enregistrement de la prédiction
        log_inference(user_input, label, confidence)
        st.caption("✅ Inférence enregistrée avec succès dans `inference_log.csv`.")

# Section de monitoring basique visible dans l'UI
if os.path.exists(LOG_FILE):
    with st.expander("📊 Consulter les derniers logs d'inférence (Monitoring)"):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            st.text(f.read()[-1000:]) # Affiche les derniers caractères du log