import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print("--- RUNNING : PHASE 1 ---")

# 1. Chargement du CSV
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=["Passengers"])

# 2. Conversion et analyse
dataset = df.values.astype('float32')
print(f"Shape initiale : {dataset.shape}")
print(f"Valeur minimale : {dataset.min()}")
print(f"Valeur maximale : {dataset.max()}\n")

# 3. Normalisation (MinMax 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

# 4. Split temporel (67% Train / 33% Test) - Pas de shuffle
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:]

# 5. Fonction de Sliding Window
def create_dataset(dataset, window_size=12):
    """Transforme une série 1D en paires (X, y) via sliding window."""
    if np.isnan(dataset).any():
        raise ValueError("Le dataset contient des valeurs NaN.")
        
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:(i + window_size), 0])
        y.append(dataset[i + window_size, 0])
        
    X_array = np.array(X)
    return np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1)), np.array(y)

# 6. Application
WINDOW_SIZE = 12
X_train, y_train = create_dataset(train, WINDOW_SIZE)
X_test, y_test = create_dataset(test, WINDOW_SIZE)

print("--- SHAPES OBTENUES ---")
print(f"X_train : {X_train.shape} | y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}  | y_test  : {y_test.shape}")