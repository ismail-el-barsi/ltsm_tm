# app.py -- À lancer avec la commande : streamlit run app.py
import streamlit as st
import numpy as np
import keras
from keras.datasets import imdb

# Importation sécurisée du padding selon l'écosystème TensorFlow/Keras de la machine
try:
    from keras.utils import pad_sequences
except ImportError:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Configuration des Hyperparamètres globaux
VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = 'imdb_sentiment.keras'

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

# =====================================================================
# 2. CHARGEMENT ET MISE EN CACHE DES RESSOURCES (Performance & Latence)
# =====================================================================
@st.cache_resource
def load_sentiment_model(model_path):
    """Charge le modèle d'inférence LSTM et le conserve en RAM."""
    return keras.models.load_model(model_path)

@st.cache_resource
def load_imdb_word_index():
    """Charge et ajuste le dictionnaire de tokens IMDB (+3 shift standard)."""
    word_index = imdb.get_word_index()
    # Décalage de 3 pour intégrer les tokens spéciaux requis par l'Embedding
    adjusted_index = {k: (v + 3) for k, v in word_index.items()}
    adjusted_index["<PAD>"] = 0
    adjusted_index["<START>"] = 1
    adjusted_index["<UNK>"] = 2
    return adjusted_index

# Chargement initial (Bloquant uniquement au tout premier démarrage de l'app)
with st.spinner("Initialisation de l'IA et chargement des poids du modèle..."):
    model = load_sentiment_model(MODEL_PATH)
    imdb_index = load_imdb_word_index()

# =====================================================================
# 3. PIPELINE DE PRÉTRAITEMENT TEXTUEL
# =====================================================================
def preprocess_text(text, word_index, max_len, vocab_size):
    """Nettoie, tokenize et padde le texte brut saisi par l'utilisateur."""
    # Nettoyage rudimentaire de la ponctuation et passage en minuscules
    clean_text = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    words = clean_text.split()
    
    # Construction de la séquence numérique
    tokens = [1] # Initialisation obligatoire avec le token <START>
    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            tokens.append(word_index[word])
        else:
            tokens.append(2) # Remplacement par <UNK> si mot rare ou inconnu
            
    # Application du padding strict en configuration 'pre' (conforme Phase 4 & 5)
    padded_sequence = pad_sequences([tokens], maxlen=max_len, padding='pre', truncating='post')
    return padded_sequence

# =====================================================================
# 4. INTERFACE UTILISATEUR STREAMLIT
# =====================================================================
st.title("🎬 Analyseur de Sentiment IMDB")
st.write("Saisissez une critique de film (en anglais). Notre modèle de Deep Learning **LSTM Bidirectionnel** déterminera si votre avis est globalement positif ou négatif.")

# Zone de saisie utilisateur
user_input = st.text_area("Votre critique de film :", height=150, placeholder="Type your movie review here...")

# Note informative pour l'adversarial case (limite technique documentée)
st.caption("⚠️ Note : les textes de plus de 200 mots sont automatiquement tronqués pour correspondre au contexte du réseau.")

# Déclenchement de l'inférence
if st.button("Analyser le sentiment", type="primary"):
    
    # Edge case check : Saisie vide ou uniquement constituée d'espaces
    if not user_input.strip():
        st.warning("Veuillez saisir du texte avant de lancer l'analyse.")
        
    else:
        # 1. Prétraitement du texte
        processed_input = preprocess_text(user_input, imdb_index, MAX_LEN, VOCAB_SIZE)
        
        # 2. Inférence (Prédiction brute)
        with st.spinner("Calcul des probabilités en cours..."):
            raw_prediction = model.predict(processed_input, verbose=0)[0][0]
        
        # 3. Post-traitement et affichage des résultats dynamiques
        st.subheader("Résultat de l'analyse")
        
        if raw_prediction >= 0.5:
            confidence = raw_prediction * 100
            st.success(f"**Sentiment : POSITIF** 🟢")
            st.metric(label="Score de confiance", value=f"{confidence:.2f}%")
        else:
            confidence = (1 - raw_prediction) * 100
            st.error(f"**Sentiment : NÉGATIF** 🔴")
            st.metric(label="Score de confiance", value=f"{confidence:.2f}%")
            
        st.info(f"Probabilité brute retournée par la couche Sigmoïde : {raw_prediction:.4f}")