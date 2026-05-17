import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import keras
from keras import layers
from keras.callbacks import EarlyStopping

print("--- RUNNING : PHASE 2 ---")

# =====================================================================
# PRÉREQUIS (Reprise rapide de la Phase 1 pour que le script fonctionne)
# =====================================================================
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
# =====================================================================

# 1. Construction du modèle LSTM
model = keras.Sequential([
    layers.LSTM(64, input_shape=(WINDOW_SIZE, 1), return_sequences=False),
    layers.Dropout(0.2), 
    layers.Dense(1)       
])

# 2. Compilation
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

model.summary()

# 3. Entraînement
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,             
    batch_size=8,           
    validation_split=0.1,   
    callbacks=[early_stopping],
    verbose=1
)

# 4. Prédictions et Dénormalisation
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict_inv = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict_inv = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. Calcul du RMSE
train_rmse = math.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
test_rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict_inv))

print("\n--- RÉSULTATS ÉVALUATION ---")
print(f"RMSE Train : {train_rmse:.2f} passagers")
print(f"RMSE Test  : {test_rmse:.2f} passagers")

# 6. Sauvegarde du modèle
model.save('airline_lstm.keras')
print("Modèle sauvegardé sous 'airline_lstm.keras'.")

# 7. Tracé de la Perte (Loss)
plt.figure(figsize=(10, 3))
plt.plot(history.history['loss'], label='Loss Train')
plt.plot(history.history['val_loss'], label='Loss Val')
plt.title('Évolution de la Perte (MSE)')
plt.legend()
plt.show()

# 8. Préparation de l'alignement pour le graphique final
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[WINDOW_SIZE:len(train_predict_inv) + WINDOW_SIZE, :] = train_predict_inv

test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
start_test_idx = len(train) + WINDOW_SIZE
test_predict_plot[start_test_idx:start_test_idx + len(test_predict_inv), :] = test_predict_inv

# 9. Graphique Final Prédictions vs Réel
plt.figure(figsize=(12, 5))
plt.plot(scaler.inverse_transform(dataset_scaled), label="Ground Truth (Réel)", color="black", alpha=0.4)
plt.plot(train_predict_plot, label="Prédictions Train", color="blue")
plt.plot(test_predict_plot, label="Prédictions Test (Futur)", color="red")
plt.title("Prédiction du Trafic Aérien par LSTM")
plt.xlabel("Mois")
plt.ylabel("Nombre de Passagers")
plt.legend()
plt.show()