import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
from keras.utils import pad_sequences  # Utilisation sécurisée pour Keras 3

print("--- RUNNING : PHASE 4 (NLP IMDB) ---")

# 1. Configuration des hyperparamètres
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBED_DIM = 128

# 2. Chargement des données IMDB (déjà tokenisées en entiers par Keras)
(X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# 3. Analyse rapide des longueurs de séquences réelles
lengths = [len(seq) for seq in X_train_raw]
print("--- Analyse des séquences brutes ---")
print(f"Nombre de reviews d'entraînement : {len(X_train_raw)}")
print(f"Longueur minimale : {np.min(lengths)} mots")
print(f"Longueur maximale : {np.max(lengths)} mots")
print(f"Longueur moyenne  : {np.mean(lengths):.1f} mots\n")

# 4. Padding / Truncating
# Choix de conception : 
# - truncating='post' pour conserver le DÉBUT de la review si elle dépasse 200 mots.
# - padding='pre' pour accumuler les zéros au début et laisser les mots importants à la fin (crucial pour le LSTM).
X_train_pad = pad_sequences(X_train_raw, maxlen=MAX_LEN, padding='pre', truncating='post')
X_test_pad = pad_sequences(X_test_raw, maxlen=MAX_LEN, padding='pre', truncating='post')

print("--- Shapes après Padding ---")
print(f"X_train_pad shape : {X_train_pad.shape}")
print(f"X_test_pad shape  : {X_test_pad.shape}\n")

# 5. Instanciation de la couche Embedding dans un modèle Sequential Keras 3
model = keras.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
])

# 6. Vérification de la transformation de shape
print("--- Summary de la couche d'Embedding ---")
model.summary()