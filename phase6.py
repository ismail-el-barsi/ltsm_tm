import numpy as np
import keras
from keras import layers
from keras.utils import pad_sequences
from datasets import load_dataset
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer 

print("--- RUNNING : PHASE 6 (Multiclass LSTM - AG News) ---")
# Le reste de ton code reste strictement identique...

# 1. Chargement des données via HuggingFace
dataset = load_dataset('ag_news')

# Extraction des structures de données textuelles et des étiquettes (0 à 3)
train_texts = dataset['train']['text']
train_labels = np.array(dataset['train']['label'])
test_texts = dataset['test']['text']
test_labels = np.array(dataset['test']['label'])

LABELS_MAP = ['World', 'Sports', 'Business', 'Sci/Tech']

print(f"Distribution initiale train : {np.bincount(train_labels)}")

# 2. Tokenization et Alignement des vocabulaires (From scratch)
VOCAB_SIZE_AG = 20000
MAX_LEN_AG = 50
EMBED_DIM_AG = 64

# Configuration du Tokenizer (fit EXCLUSIVEMENT sur le train set pour éviter le data leakage)
tokenizer_ag = Tokenizer(num_words=VOCAB_SIZE_AG, oov_token='<OOV>')
tokenizer_ag.fit_on_texts(train_texts)

# Transformation des textes bruts en listes d'entiers paddées
X_train_ag = pad_sequences(tokenizer_ag.texts_to_sequences(train_texts), maxlen=MAX_LEN_AG, padding='pre', truncating='post')
X_test_ag = pad_sequences(tokenizer_ag.texts_to_sequences(test_texts), maxlen=MAX_LEN_AG, padding='pre', truncating='post')

print(f"Shape finale X_train_ag : {X_train_ag.shape}") # (120000, 50)
print(f"Shape finale X_test_ag  : {X_test_ag.shape}\n")  # (7600, 50)

# 3. Architecture du Réseau Multiclasse
model_ag = keras.Sequential([
    layers.Input(shape=(MAX_LEN_AG,)),
    layers.Embedding(input_dim=VOCAB_SIZE_AG, output_dim=EMBED_DIM_AG),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    # Sortie à 4 neurones couplée au Softmax pour obtenir une distribution probabiliste (somme = 1)
    layers.Dense(4, activation='softmax')
])

# Utilisation de la 'sparse_categorical_crossentropy' car les labels cibles sont des entiers (0, 1, 2, 3) 
# et non des vecteurs one-hot encodes.
model_ag.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_ag.summary()

# 4. Entraînement (Le dataset est large, 3 epochs suffisent à converger)
history_ag = model_ag.fit(
    X_train_ag, train_labels,
    epochs=3,
    batch_size=256,
    validation_split=0.05,
    verbose=1
)

# 5. Évaluation globale et Rapport de performance
loss_ag, acc_ag = model_ag.evaluate(X_test_ag, test_labels, verbose=0)
print(f"\nAccuracy globale AG News sur le Test Set : {acc_ag:.4f}")

y_pred_proba = model_ag.predict(X_test_ag, verbose=0)
# Extraction de l'index de classe ayant obtenu la probabilité maximale (axe des colonnes)
y_pred_ag = np.argmax(y_pred_proba, axis=1)

print("\nRapport de Classification :")
print(classification_report(test_labels, y_pred_ag, target_names=LABELS_MAP))


# ==========================================
# 6. QUALITY GATE AUTOMATISÉ
# ==========================================
print("\n" + "="*15 + " QUALITY GATE : ADVERSARIAL " + "="*15)

def predict_ag_news(text, tokenizer, model, max_len, labels):
    """Tokenize un texte personnalisé et affiche la distribution de probabilité par classe."""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='pre', truncating='post')
    proba = model.predict(padded, verbose=0)[0]
    
    print(f"\nTitre évalué : \"{text}\"")
    for i, label_name in enumerate(labels):
        print(f"  -> {label_name:<10} : {proba[i]*100:.2f}%")

# Test sur les titres ambigus à cheval sur plusieurs catégories
predict_ag_news("Tesla announces record quarterly profits", tokenizer_ag, model_ag, MAX_LEN_AG, LABELS_MAP)
predict_ag_news("Olympics 2024: Tech companies sponsor all major sports events", tokenizer_ag, model_ag, MAX_LEN_AG, LABELS_MAP)
print("="*44)