import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Chargement des données
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=["Passengers"])

# 2. Conversion en float32 et analyse rapide
dataset = df.values.astype('float32')
print("--- Analyse Initiale ---")
print(f"Shape initiale : {dataset.shape}")
print(f"Valeur minimale : {dataset.min()}")
print(f"Valeur maximale : {dataset.max()}\n")

# 3. Normalisation entre 0 et 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# 4. Split temporel chronologique (67% Train / 33% Test) - Aucun shuffle !
train_size = int(len(dataset_scaled) * 0.67)
test_size = len(dataset_scaled) - train_size
train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:len(dataset_scaled)]

# 5. Fonction de Sliding Window (Fenêtre Glissante)
def create_dataset(dataset, window_size=12):
    """
    Transforme une série 1D en paires (X, y) via sliding window.
    Retourne X : (N - window_size, window_size, 1), y : (N - window_size,)
    """
    # Protection Adversarial : Vérification de la présence de NaNs
    if np.isnan(dataset).any():
        raise ValueError("Le dataset contient des valeurs NaN. Veuillez nettoyer les données avant de segmenter.")
        
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:(i + window_size), 0])
        y.append(dataset[i + window_size, 0])
        
    # Conversion en tableaux numpy et ajout de la dimension finale pour le LSTM (features)
    X_array = np.array(X)
    X_array = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1))
    
    return X_array, np.array(y)

# 6. Application de la fonction
WINDOW_SIZE = 12
X_train, y_train = create_dataset(train, WINDOW_SIZE)
X_test, y_test = create_dataset(test, WINDOW_SIZE)

print("--- Shapes Après Sliding Window ---")
print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")