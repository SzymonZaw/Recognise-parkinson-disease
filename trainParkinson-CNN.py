import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from joblib import dump
import matplotlib.pyplot as plt

# Definicja zoptymalizowanej sieci neuronowej
class OptimizedNN(nn.Module):
    def __init__(self, input_size):
        super(OptimizedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)  # Dropout do regularyzacji
        self.fc4 = nn.Linear(64, 1)  # Wyjście modelu (jedna wartość predykcyjna)

    def forward(self, x):
        # Przepływ danych przez warstwy sieci
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_with_cross_validation(X, y, part_name, n_splits=5, num_epochs=100):
    kf = KFold(n_splits=n_splits)  # Kroswalidacja z n_splits podziałami
    all_scores = []  # Lista do przechowywania wyników MSE
    input_size = X.shape[1]  # Rozmiar wejścia (liczba cech)

    for train_index, test_index in kf.split(X):
        # Przekształcanie danych na tensory PyTorch
        X_train, X_test = torch.tensor(X[train_index], dtype=torch.float32), torch.tensor(X[test_index], dtype=torch.float32)
        y_train, y_test = torch.tensor(y[train_index], dtype=torch.float32), torch.tensor(y[test_index], dtype=torch.float32)

        # Inicjalizacja modelu i optymalizatora
        model = OptimizedNN(input_size)
        criterion = nn.MSELoss()  # Funkcja kosztu (MSE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optymalizator Adam

        # Przygotowanie danych do treningu
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Dataloader do mini-batchy

        # Trenowanie modelu
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Ewaluacja modelu
        model.eval()
        y_pred = model(X_test).squeeze().detach().numpy()  # Predykcje na danych testowych
        mse = mean_squared_error(y_test, y_pred)  # Obliczanie błędu średniokwadratowego
        all_scores.append(mse)

    avg_score = np.mean(all_scores)  # Średni wynik MSE z kroswalidacji
    print(f'Kroswalidacja {part_name} - Średnie MSE: {avg_score:.4f}')

    return model  # Zwrócenie wytrenowanego modelu

# Funkcja do analizy wyników modelu
def analyze_results(y_true, y_pred):
    residuals = y_true - y_pred  # Reszty predykcji
    plt.hist(residuals, bins=50)  # Histogram reszt
    plt.title('Histogram reszt')

    plt.scatter(y_true, y_pred)  # Wykres rzeczywistych wartości vs. przewidywanych
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r')  # Linia odniesienia
    plt.xlabel('Rzeczywiste wartości')
    plt.ylabel('Przewidywane wartości')
    plt.title('Rzeczywiste vs Przewidywane')

# Funkcja do trenowania i walidacji modeli dla każdej części ciała
def train_and_evaluate(X, y, part_name):
    scaler = MinMaxScaler(feature_range=(0, 3))  # Skalowanie cech do przedziału [0, 3]
    X_scaled = scaler.fit_transform(X.values)  # Normalizacja danych
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Kroswalidacja
    model = train_with_cross_validation(X_tensor.numpy(), y_tensor.numpy(), part_name)

    # Ostateczna ewaluacja modelu na pełnym zestawie testowym
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    model.eval()
    y_pred = model(X_test).squeeze().detach().numpy()
    mse = mean_squared_error(y_test, y_pred)  # Błąd średniokwadratowy
    r2 = r2_score(y_test, y_pred)  # Wskaźnik R2
    print(f'{part_name} - Ostateczna ewaluacja: MSE: {mse:.4f}, R2: {r2:.4f}')

    # Analiza wyników
    analyze_results(y_test.numpy(), y_pred)

    return model, scaler

# Wczytywanie danych komunikacji z pliku
trajectory_file = 'Communication_all_export.txt'
with open(trajectory_file, 'r') as infile:
    comm_dict = json.load(infile)  # Słownik danych komunikacji

# Wczytywanie ocen UDysRS z pliku
rating_file = 'UDysRS.txt'
with open(rating_file, 'r') as infile:
    ratings = json.load(infile)  # Słownik ocen

# Funkcja do ekstrakcji cech z danych pozycji
def extract_features(positions):
    features = {}
    for joint, coords in positions.items():
        if len(coords) > 0:
            coords = np.array(coords)
            # Obliczanie podstawowych statystyk
            mean_x = np.mean(coords[:, 0])
            mean_y = np.mean(coords[:, 1])
            std_x = np.std(coords[:, 0])
            std_y = np.std(coords[:, 1])
            features[f'{joint}_mean_x'] = mean_x
            features[f'{joint}_mean_y'] = mean_y
            features[f'{joint}_std_x'] = std_x
            features[f'{joint}_std_y'] = std_y
            features[f'{joint}_max_x'] = np.max(coords[:, 0])
            features[f'{joint}_min_x'] = np.min(coords[:, 0])
            features[f'{joint}_max_y'] = np.max(coords[:, 1])
            features[f'{joint}_min_y'] = np.min(coords[:, 1])

            # Magnituda FFT
            fft_x = np.abs(fft(coords[:, 0]))  # Magnituda FFT dla współrzędnych x
            fft_y = np.abs(fft(coords[:, 1]))  # Magnituda FFT dla współrzędnych y
            features[f'{joint}_fft_magnitude'] = np.mean(fft_x) + np.mean(fft_y)  # Agregacja magnitudy

            # Wariancja
            variance_x = np.var(coords[:, 0])
            variance_y = np.var(coords[:, 1])
            features[f'{joint}_variance_x'] = variance_x
            features[f'{joint}_variance_y'] = variance_y

            # Kurtosis
            kurtosis_x = kurtosis(coords[:, 0])
            kurtosis_y = kurtosis(coords[:, 1])
            features[f'{joint}_kurtosis_x'] = kurtosis_x
            features[f'{joint}_kurtosis_y'] = kurtosis_y

            # Skewness
            skewness_x = skew(coords[:, 0])
            skewness_y = skew(coords[:, 1])
            features[f'{joint}_skewness_x'] = skewness_x
            features[f'{joint}_skewness_y'] = skewness_y

        else:
            # Wartości NaN dla pustych danych
            features[f'{joint}_mean_x'] = features[f'{joint}_mean_y'] = np.nan
            features[f'{joint}_std_x'] = features[f'{joint}_std_y'] = np.nan
            features[f'{joint}_max_x'] = features[f'{joint}_min_x'] = np.nan
            features[f'{joint}_max_y'] = features[f'{joint}_min_y'] = np.nan
            features[f'{joint}_fft_magnitude'] = np.nan
            features[f'{joint}_variance_x'] = features[f'{joint}_variance_y'] = np.nan
            features[f'{joint}_kurtosis_x'] = features[f'{joint}_kurtosis_y'] = np.nan
            features[f'{joint}_skewness_x'] = features[f'{joint}_skewness_y'] = np.nan

    return features

# Przetwarzanie danych komunikacji
comm_data = []
for key, data in comm_dict.items():
    trial = key.split()[0].split('-')[0]  # Wyciąganie numeru próby
    positions = data['position']
    features = extract_features(positions)
    features['Trial'] = trial  # Dodanie numeru próby do cech
    comm_data.append(features)

# Konwersja do DataFrame
comm_df = pd.DataFrame(comm_data)

# Przetwarzanie ocen UDysRS
ratings_data = []
for trial_key, scores in ratings['Communication'].items():
    ratings_data.append([trial_key] + scores)  # Dodanie ocen do danych

udysrs_columns = ['Trial', 'Neck', 'Right_arm_shoulder', 'Left_arm_shoulder', 'Trunk', 'Right_leg_hip', 'Left_leg_hip']
ratings_df = pd.DataFrame(ratings_data, columns=udysrs_columns)

# Łączenie danych komunikacji z ocenami
merged_df = pd.merge(comm_df, ratings_df, on='Trial')

# Trenowanie modeli dla każdej części ciała
models = {}
scalers = {}
for part in udysrs_columns[1:]:
    X = merged_df.drop(columns=['Trial'] + udysrs_columns[1:])  # Cechy (bez kolumn z ocenami)
    y = merged_df[part]  # Oceny dla danej części ciała
    model, scaler = train_and_evaluate(X, y, part)  # Trenowanie i walidacja modelu
    models[part] = model
    scalers[part] = scaler

# Zapis modeli i skalera
os.makedirs('models', exist_ok=True)  # Utworzenie katalogu na modele, jeśli nie istnieje
for part, model in models.items():
    torch.save(model.state_dict(), f'models/{part}_model.pth')  # Zapis modelu
    dump(scalers[part], f'models/{part}_scaler.joblib')  # Zapis skalera w formacie .joblib
