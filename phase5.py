import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, classification_report

print("--- RUNNING : PHASE 5 (Bi-LSTM IMDB) ---")

# ==========================================
# 1. CONSTANTES & PRÉPARATION (Phase 4)
# ==========================================
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBED_DIM = 128

(X_train_raw, y_train_imdb), (X_test_raw, y_test_imdb) = imdb.load_data(num_words=VOCAB_SIZE)

# Padding 'pre' et truncating 'post' conformes aux spécifications de la phase 4
X_train_pad = pad_sequences(X_train_raw, maxlen=MAX_LEN, padding='pre', truncating='post')
X_test_pad = pad_sequences(X_test_raw, maxlen=MAX_LEN, padding='pre', truncating='post')

# ==========================================
# 2. ARCHITECTURE DU MODÈLE BI-LSTM
# ==========================================
model_imdb = keras.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    # Le wrapper Bidirectional traite la séquence dans les deux sens (Direct / Inverse)
    # Concatène les hidden states : l'output effectif en sortie sera de 2 x 64 = 128 dimensions.
    layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
    layers.Dropout(0.5), # Taux fort pour contrer le bruit et le surapprentissage
    layers.Dense(1, activation='sigmoid') # Classification binaire
])

model_imdb.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_imdb.summary()

# ==========================================
# 3. ENTRAÎNEMENT
# ==========================================
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=3, 
    restore_best_weights=True
)

history_imdb = model_imdb.fit(
    X_train_pad, y_train_imdb,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 4. ÉVALUATION STANDARD
# ==========================================
loss_test, acc_test = model_imdb.evaluate(X_test_pad, y_test_imdb, verbose=0)
print("\n" + "="*40)
print(f"Accuracy globale test : {acc_test:.4f}")

# Prédiction et binarisation au seuil de 0.5
y_pred_proba = model_imdb.predict(X_test_pad, verbose=0)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

# Calcul du F1-score
f1 = f1_score(y_test_imdb, y_pred)
print(f"F1-score global test  : {f1:.4f}")
print("="*40)

print("\nRapport de classification complet :")
print(classification_report(y_test_imdb, y_pred, target_names=['Négatif', 'Positif']))

# Sauvegarde obligatoire du modèle pour Keras 3
model_imdb.save('imdb_sentiment.keras')
print("Modèle IMDB sauvegardé sous 'imdb_sentiment.keras'.\n")


# ==========================================
# 5. QUALITY GATE AUTOMATISÉ
# ==========================================
print("\n" + "="*15 + " QUALITY GATE " + "="*15)

# --- Edge Case : Évaluation des séquences ultra-courtes (< 10 mots) ---
short_indices = [i for i, seq in enumerate(X_test_raw) if len(seq) < 10]
if len(short_indices) > 0:
    X_test_short = X_test_pad[short_indices]
    y_test_short = y_test_imdb[short_indices]
    
    loss_short, acc_short = model_imdb.evaluate(X_test_short, y_test_short, verbose=0)
    print(f"[Edge Case] Accuracy sur les avis très courts (< 10 mots) : {acc_short:.4f} (Basé sur {len(short_indices)} avis)")
else:
    print("[Edge Case] Aucun avis de moins de 10 mots trouvé dans le jeu de test.")

# --- Adversarial Case : Test sur des reviews complexes ---
def predict_custom_review(text_review):
    word_index = imdb.get_word_index()
    # Alignement du dictionnaire avec le décalage standard d'IMDB Keras (+3)
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    
    # Tokenization de base
    words = text_review.lower().replace(",", "").replace(".", "").split()
    tokens = [1] # Start token
    for word in words:
        if word in word_index and word_index[word] < VOCAB_SIZE:
            tokens.append(word_index[word])
        else:
            tokens.append(2) # UNK (Out of vocabulary)
            
    # Padding
    padded_tokens = pad_sequences([tokens], maxlen=MAX_LEN, padding='pre', truncating='post')
    
    # Inférence
    proba = model_imdb.predict(padded_tokens, verbose=0)[0][0]
    sentiment = "Positif" if proba >= 0.5 else "Négatif"
    print(f"Review : \"{text_review}\"")
    print(f"-> Score de positivité : {proba:.4f} | Sentiment détecté : {sentiment}\n")

print("\n[Adversarial Tests] Inférences sur des textes spécifiques :")
# Review brute négative
predict_custom_review("Absolutely terrible, boring beyond belief, waste of time")
# Reviews sarcastiques / ambiguës
predict_custom_review("Truly great movie... if you look for a perfect way to fall asleep quickly")
predict_custom_review("A masterpiece of boredom, I highly recommend it to anyone who loves to suffer")
print("="*44)