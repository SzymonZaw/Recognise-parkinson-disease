import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import os
from joblib import load
import pandas as pd

# Definicja zoptymalizowanej sieci neuronowej do przewidywania stopnia Parkinsonizmu
class OptimizedNN(nn.Module):
    def __init__(self, input_size):
        super(OptimizedNN, self).__init__()
        # Warstwy w pełni połączone (fully connected) z dropoutem
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)  # Dropout dla zapobiegania nadmiernemu dopasowaniu (overfitting)
        self.fc4 = nn.Linear(64, 1)  # Wyjście jednowymiarowe (pojedynczy stopień zaawansowania choroby)

    def forward(self, x):
        # Funkcje aktywacji ReLU dla ukrytych warstw
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout, aby losowo wyłączać neurony podczas trenowania
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Brak funkcji aktywacji na wyjściu (bo mamy regresję)
        return x

# Funkcja do załadowania danych kluczowych punktów ciała (keypoints) z pliku JSON
def load_predicted_keypoints(file_content):
    try:
        # Próba wczytania danych z pliku w formacie JSON
        data = json.loads(file_content)
    except json.JSONDecodeError:
        st.error("Błąd wczytywania danych JSON.")  # Komunikat o błędzie, jeśli dane są niepoprawne
        return None

    # Zdefiniowana kolejność kluczowych punktów ciała, które będziemy używać
    keypoints_order = [
        "head", "neck", "Lkne", "Lwri", "Rkne", "Lelb", "Lsho",
        "Rhip", "Rank", "face", "Lhip", "Rwri", "Lank", "Relb", "Rsho"
    ]

    features = []  # Lista na przetworzone cechy
    for keypoint in keypoints_order:
        # Dla każdego punktu kluczowego zbieramy różne statystyki
        for stat in ['mean_x', 'mean_y', 'std_x', 'std_y', 'max_x', 'min_x', 'max_y', 'min_y', 'fft_magnitude', 'variance_x', 'variance_y', 'kurtosis_x', 'kurtosis_y', 'skewness_x', 'skewness_y']:
            feature_value = data.get(keypoint, {}).get(stat, np.nan)  # Pobieramy wartość statystyki, jeśli istnieje
            if isinstance(feature_value, float) and np.isnan(feature_value):
                features.append(0)  # Zamiana wartości NaN na 0, jeśli nie ma danych
            else:
                features.append(feature_value)  # Dodanie wartości do listy cech

    # Konwertowanie listy cech na tablicę NumPy
    features_array = np.array(features).reshape(1, -1)

    # Sprawdzanie, czy mamy za dużo brakujących danych (NaN)
    if np.isnan(features_array).sum() > 0.5 * features_array.size:
        st.error("Zbyt wiele brakujących danych, nie można kontynuować.")  # Błąd, jeśli jest zbyt wiele braków
        return None

    return features_array  # Zwrócenie przetworzonych cech

# Funkcja do ładowania modelu dla danej części ciała
def load_model(part_name, input_size):
    sanitized_part = part_name.replace(" ", "_").replace("/", "_")  # Zamiana spacji i znaków specjalnych w nazwie
    model_path = os.path.join('models', f'{sanitized_part}_model.pth')  # Ścieżka do modelu

    model = OptimizedNN(input_size)  # Inicjalizacja modelu
    try:
        model.load_state_dict(torch.load(model_path))  # Ładowanie wag modelu z pliku
        model.eval()  # Ustawienie modelu w tryb ewaluacji
        return model
    except FileNotFoundError:
        st.error(f"Nie znaleziono modelu dla {part_name}.")  # Błąd, jeśli plik modelu nie istnieje
    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu dla {part_name}: {e}")  # Błąd podczas ładowania
    return None  # Zwraca None, jeśli nie udało się załadować modelu

# Funkcja do zaokrąglania w krokach co 0.5
def round_custom(value):
    return round(value * 2) / 2  # Mnożenie przez 2, zaokrąglanie do najbliższej liczby całkowitej i dzielenie przez 2



# Funkcja do przewidywania wyniku przy użyciu modelu
def predict_result(model, data):
    model.eval()  # Ustawienie modelu w tryb ewaluacji (wyłączenie dropoutu)
    with torch.no_grad():  # Wyłączenie gradientów, bo nie trenujemy modelu
        inputs = torch.tensor(data, dtype=torch.float32)  # Konwertowanie danych wejściowych na tensor
        if inputs.dim() != 2:
            raise ValueError(f"Nieprawidłowy wymiar danych wejściowych: {inputs.dim()}. Oczekiwano 2D.")  # Sprawdzenie wymiaru danych wejściowych
        if inputs.shape[1] != model.fc1.in_features:
            raise ValueError(f"Liczba cech ({inputs.shape[1]}) nie zgadza się z wejściową liczbą cech modelu ({model.fc1.in_features}).")  # Sprawdzenie zgodności liczby cech

        inputs = torch.nan_to_num(inputs)  # Zamiana NaN na 0 w danych wejściowych

        output = model(inputs).item()  # Przewidywanie wyniku (stopnia zaawansowania choroby)

        # Ograniczanie wyniku do przedziału [0, 3]
        output = max(0, min(3, output))

        # Zaokrąglanie wyniku w krokach co 0.5
        output = round_custom(output)

        return output  # Zwrócenie przewidywanego stopnia zaawansowania



# Główna funkcja do przeprowadzenia treningu i predykcji dla choroby Parkinsona
def parkinsonTrain(file_content):
    if file_content:
        # Ładowanie danych z pliku kluczowych punktów ciała
        predicted_data = load_predicted_keypoints(file_content)

        if predicted_data is None or predicted_data.size == 0:
            st.error("Brak prawidłowych danych do przewidywań.")  # Sprawdzenie, czy dane są prawidłowe
            return

        input_size = predicted_data.shape[1]  # Ustal liczbę cech na podstawie danych wejściowych

        # Mapowanie punktów kluczowych na części ciała
        keypoint_to_part = {
            "Lsho": "Left_arm_shoulder",
            "Rhip": "Right_leg_hip",
            "Lhip": "Left_leg_hip",
            "Rsho": "Right_arm_shoulder"
        }

        models = {}  # Słownik na modele
        scalers = {}  # Słownik na scalery

        # Dla każdej części ciała ładowanie odpowiedniego modelu i skalera
        for part in set(keypoint_to_part.values()):
            model = load_model(part, input_size)
            if model is None:
                return

            sanitized_part = part.replace(" ", "_").replace("/", "_")
            scaler_path = os.path.join('models', f'{sanitized_part}_scaler.joblib')
            try:
                scaler = load(scaler_path)  # Ładowanie skalera do normalizacji danych
            except FileNotFoundError:
                st.error(f"Nie znaleziono skalera dla {part}.")  # Błąd, jeśli skaler nie istnieje
                return

            models[part] = model
            scalers[part] = scaler

        predictions = {}  # Słownik na wyniki predykcji
        # Dla każdego punktu kluczowego dokonywanie predykcji przy użyciu modelu
        for keypoint, part in keypoint_to_part.items():
            model = models.get(part)
            scaler = scalers.get(part)
            if model and scaler:
                # Normalizacja danych przy użyciu skalera
                features_scaled = scaler.transform(predicted_data)

                # Dokonywanie predykcji przy użyciu modelu
                prediction_value = predict_result(model, features_scaled)
                predictions[keypoint] = prediction_value  # Zapisanie wyniku

        # Wyświetlanie wyników predykcji
        for keypoint, prediction in predictions.items():
            st.write(f"Przewidywany stopień Parkinsonizmu dla {keypoint}: {prediction:.2f}")

        # Przygotowanie wyników do pobrania jako plik CSV
        df_predictions = pd.DataFrame(list(predictions.items()), columns=['Punkt kluczowy', 'Przewidywany stopień'])
        csv_data = df_predictions.to_csv(index=False)
        st.download_button(
            label="Pobierz wyniki jako CSV",
            data=csv_data,
            file_name="wyniki.csv",
            mime="text/csv"
        )
    else:
        st.info("Proszę wprowadzić dane JSON, aby kontynuować.")  # Informacja, jeśli nie ma danych JSON
