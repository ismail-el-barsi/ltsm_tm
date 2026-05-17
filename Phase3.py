import numpy as np
import pandas as pd
import math
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import keras
from keras import layers
from keras.callbacks import EarlyStopping

print("--- RUNNING : PHASE 3 (LSTM vs GRU) ---")

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=["Passengers"])
dataset = df.values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[0:train_size], dataset_scaled[train_size:]

def create_dataset(dataset, window_size=12):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:(i + window_size), 0])
        y.append(dataset[i + window_size, 0])
    return np.reshape(np.array(X), (len(X), window_size, 1)), np.array(y)

WINDOW_SIZE = 12
X_train, y_train = create_dataset(train, WINDOW_SIZE)
X_test, y_test = create_dataset(test, WINDOW_SIZE)

# ==========================================
# 2. FONCTION DE BENCHMARK CORRIGÉE
# ==========================================
def build_and_train(rnn_layer_class, units=64, epochs=50, label=''):
    """
    Construit et entraîne un modèle avec la couche RNN passée en paramètre (LSTM ou GRU).
    """
    print(f"\n=== Entraînement du modèle : {label} ===")
    
    # Construction propre conforme à Keras 3 (utilisation de layers.Input)
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 1)),
        rnn_layer_class(units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    # Compilation
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Arrêt précoce
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Mesure du temps
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=8,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0  # Silencieux pour éviter de polluer le terminal
    )
    end_time = time.time()
    
    total_time = end_time - start_time
    actual_epochs = len(history.history['loss'])
    time_per_epoch = total_time / actual_epochs if actual_epochs > 0 else 0
    
    # Prédiction et dénormalisation
    test_predict = model.predict(X_test, verbose=0)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calcul des métriques
    rmse_test = math.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    val_loss_finale = history.history['val_loss'][-1]
    num_params = model.count_params()
    
    # Affichage des résultats
    print(f"-> Nombre de paramètres : {num_params}")
    print(f"-> Epochs exécutées      : {actual_epochs}/{epochs}")
    print(f"-> Temps total d'exec   : {total_time:.2f} secondes")
    print(f"-> Durée moyenne / epoch : {time_per_epoch:.4f} secondes")
    print(f"-> Val Loss finale       : {val_loss_finale:.6f}")
    print(f"-> RMSE (Test set)       : {rmse_test:.2f} passagers")
    
    # Correction de la coquille ici : rmse_test au lieu de rme_test
    return model, history, total_time, time_per_epoch, num_params, val_loss_finale, rmse_test

# ==========================================
# 3. EXÉCUTION DU COMPARATIF
# ==========================================
print("\n--- LANCEMENT DU COMPARATIF ---")

lstm_results = build_and_train(layers.LSTM, units=64, epochs=50, label='LSTM')
gru_results = build_and_train(layers.GRU, units=64, epochs=50, label='GRU')

print("\n--- TOUT EST OK ! ---")