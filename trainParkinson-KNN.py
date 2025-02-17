import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor  # Importowanie K-NN
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import kurtosis, skew

# Funkcja do trenowania modelu z wykorzystaniem kroswalidacji
def train_with_cross_validation(X, y, part_name, n_splits=5):
    kf = KFold(n_splits=n_splits)
    all_scores = []
    best_model = None
    best_score = float('inf')

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inicjalizacja i trenowanie modelu K-NN
        model = KNeighborsRegressor(n_neighbors=5)  # Użycie K-NN z 5 najbliższymi sąsiadami
        model.fit(X_train, y_train)

        # Ewaluacja modelu
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        all_scores.append(mse)

        # Zapamiętanie najlepszego modelu
        if mse < best_score:
            best_score = mse
            best_model = model

    avg_score = np.mean(all_scores)
    print(f'Kroswalidacja {part_name} - Średnie MSE: {avg_score:.4f}')

    return best_model, avg_score

# Funkcja do analizy wyników modelu
def analyze_results(y_true, y_pred):
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50)
    plt.title('Histogram reszt')
    plt.show()

    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r')
    plt.xlabel('Rzeczywiste wartości')
    plt.ylabel('Przewidywane wartości')
    plt.title('Rzeczywiste vs Przewidywane')
    plt.show()

# Funkcja do trenowania i walidacji modeli dla każdej części ciała
def train_and_evaluate(X, y, part_name):
    scaler = MinMaxScaler(feature_range=(0, 3))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skalowanie danych
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Kroswalidacja
    model, cross_val_mse = train_with_cross_validation(X_train_scaled, y_train, part_name)

    # Ostateczna ewaluacja modelu
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{part_name} - Ostateczna ewaluacja: MSE: {mse:.4f}, R2: {r2:.4f}')

    # Analiza wyników
    analyze_results(y_test, y_pred)

    return model, scaler, cross_val_mse, mse, r2

# Funkcja do ekstrakcji cech z danych pozycji
def extract_features(positions):
    features = {}
    for joint, coords in positions.items():
        if len(coords) > 0:
            coords = np.array(coords)
            features[f'{joint}_mean_x'] = np.mean(coords[:, 0])
            features[f'{joint}_mean_y'] = np.mean(coords[:, 1])
            features[f'{joint}_std_x'] = np.std(coords[:, 0])
            features[f'{joint}_std_y'] = np.std(coords[:, 1])
            features[f'{joint}_max_x'] = np.max(coords[:, 0])
            features[f'{joint}_min_x'] = np.min(coords[:, 0])
            features[f'{joint}_max_y'] = np.max(coords[:, 1])
            features[f'{joint}_min_y'] = np.min(coords[:, 1])

            # Magnituda FFT
            fft_x = np.abs(fft(coords[:, 0]))
            fft_y = np.abs(fft(coords[:, 1]))
            features[f'{joint}_fft_magnitude'] = np.mean(fft_x) + np.mean(fft_y)

            # Wariancja, skośność i kurtoza
            features[f'{joint}_variance_x'] = np.var(coords[:, 0])
            features[f'{joint}_variance_y'] = np.var(coords[:, 1])
            features[f'{joint}_kurtosis_x'] = kurtosis(coords[:, 0])
            features[f'{joint}_kurtosis_y'] = kurtosis(coords[:, 1])
            features[f'{joint}_skewness_x'] = skew(coords[:, 0])
            features[f'{joint}_skewness_y'] = skew(coords[:, 1])
        else:
            # Wartości NaN dla pustych danych
            for metric in ['mean', 'std', 'max', 'min', 'fft_magnitude', 'variance', 'kurtosis', 'skewness']:
                features[f'{joint}_{metric}_x'] = features[f'{joint}_{metric}_y'] = np.nan

    return features

# Wczytywanie danych komunikacji z pliku
trajectory_file = 'Communication_all_export.txt'
with open(trajectory_file, 'r') as infile:
    comm_dict = json.load(infile)

# Wczytywanie ocen UDysRS z pliku
rating_file = 'UDysRS.txt'
with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

# Przetwarzanie danych komunikacji
comm_data = []
for key, data in comm_dict.items():
    trial = key.split()[0].split('-')[0]
    positions = data['position']
    features = extract_features(positions)
    features['Trial'] = trial
    comm_data.append(features)

comm_df = pd.DataFrame(comm_data)

# Przetwarzanie ocen UDysRS
ratings_data = []
for trial_key, scores in ratings['Communication'].items():
    ratings_data.append([trial_key] + scores)

udysrs_columns = ['Trial', 'Neck', 'Right_arm_shoulder', 'Left_arm_shoulder', 'Trunk', 'Right_leg_hip', 'Left_leg_hip']
ratings_df = pd.DataFrame(ratings_data, columns=udysrs_columns)

# Łączenie danych komunikacji z ocenami
merged_df = pd.merge(comm_df, ratings_df, on='Trial')

# Trenowanie modeli dla każdej części ciała
results = []
for part in udysrs_columns[1:]:
    X = merged_df.drop(columns=['Trial'] + udysrs_columns[1:])
    y = merged_df[part]
    model, scaler, cross_val_mse, test_mse, test_r2 = train_and_evaluate(X.values, y.values, part)
    results.append((part, cross_val_mse, test_mse, test_r2))

# Wyświetlanie wyników
print("\nPodsumowanie wyników:")
print("Część ciała | Cross-validation MSE | Test MSE | Test R2")
for part, cross_val_mse, test_mse, test_r2 in results:
    print(f"{part:<12} | {cross_val_mse:.4f}             | {test_mse:.4f}   | {test_r2:.4f}")
