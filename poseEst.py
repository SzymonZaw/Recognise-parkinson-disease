import cv2
import mediapipe as mp
import numpy as np
import json
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
from collections import deque

# Inicjalizacja modułu MediaPipe do detekcji postawy
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Nazwy punktów kluczowych używanych przez MediaPipe Pose
keypoint_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Mapowanie nazw części ciała na indeksy punktów kluczowych
body_parts_mapping = {
    "head": [keypoint_names.index("nose")],
    "neck": [keypoint_names.index("left_shoulder"), keypoint_names.index("right_shoulder")],
    "Lkne": [keypoint_names.index("left_knee")],
    "Lwri": [keypoint_names.index("left_wrist")],
    "Rkne": [keypoint_names.index("right_knee")],
    "Lelb": [keypoint_names.index("left_elbow")],
    "Lsho": [keypoint_names.index("left_shoulder")],
    "Rhip": [keypoint_names.index("right_hip")],
    "Rank": [keypoint_names.index("right_ankle")],
    "face": [keypoint_names.index("nose"), keypoint_names.index("left_eye"), keypoint_names.index("right_eye")],
    "Lhip": [keypoint_names.index("left_hip")],
    "Rwri": [keypoint_names.index("right_wrist")],
    "Lank": [keypoint_names.index("left_ankle")],
    "Relb": [keypoint_names.index("right_elbow")],
    "Rsho": [keypoint_names.index("right_shoulder")]
}

# Parametry do monitorowania stabilności
STABILITY_THRESHOLD = 5  # liczba klatek, przez które położenie musi być stabilne
STABILITY_MARGIN = 7.0  # margines tolerancji dla stabilności

def filter_signal(data, cutoff=0.1, fs=30.0, order=5):
    """ Filtracja dolnoprzepustowa sygnału przy użyciu filtru Butterwortha """
    nyquist = 0.5 * fs  # Częstotliwość Nyquista
    normal_cutoff = cutoff / nyquist  # Normalizacja częstotliwości odcięcia
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Współczynniki filtru
    return filtfilt(b, a, data)  # Zastosowanie filtru do danych

def calculate_neck(predicted_keypoints):
    # Pobranie punktów kluczowych dla lewego i prawego barku oraz nosa
    left_shoulder = predicted_keypoints[keypoint_names.index("left_shoulder")]
    right_shoulder = predicted_keypoints[keypoint_names.index("right_shoulder")]
    nose = predicted_keypoints[keypoint_names.index("nose")]

    # Obliczanie środka barków
    shoulder_center = np.mean([left_shoulder, right_shoulder], axis=0)

    # Wyznaczenie wektora od środka barków do nosa
    neck_vector = np.array(nose) - np.array(shoulder_center)

    # Określenie pozycji szyi jako 30% odległości od środka barków do nosa
    neck_position = shoulder_center + 0.3 * neck_vector

    return neck_position

def get_statistics(keypoints):
    """Funkcja do obliczania statystyk drgań z uwzględnieniem wartości NaN"""
    keypoints = np.array(keypoints)
    x = keypoints[:, 0]
    y = keypoints[:, 1]

    valid_x = x[~np.isnan(x)]  # Filtracja wartości nie-NaN dla współrzędnych x
    valid_y = y[~np.isnan(y)]  # Filtracja wartości nie-NaN dla współrzędnych y

    # Sprawdzenie, czy wystarczająca liczba punktów jest dostępna
    if len(valid_x) < 0.7 * len(x) or len(valid_y) < 0.7 * len(y):
        return {
            'mean_x': np.nan,
            'mean_y': np.nan,
            'std_x': np.nan,
            'std_y': np.nan,
            'max_x': np.nan,
            'min_x': np.nan,
            'max_y': np.nan,
            'min_y': np.nan,
            'fft_magnitude': np.nan,
            'variance_x': np.nan,
            'variance_y': np.nan,
            'kurtosis_x': np.nan,
            'kurtosis_y': np.nan,
            'skewness_x': np.nan,
            'skewness_y': np.nan
        }

    # Filtracja sygnałów współrzędnych
    x_filtered = filter_signal(valid_x)
    y_filtered = filter_signal(valid_y)

    # Obliczanie statystyk dla współrzędnych x i y
    mean_x = np.mean(x_filtered)
    mean_y = np.mean(y_filtered)
    std_x = np.std(x_filtered, ddof=0)
    std_y = np.std(y_filtered, ddof=0)
    max_x = np.max(x_filtered)
    min_x = np.min(x_filtered)
    max_y = np.max(y_filtered)
    min_y = np.min(y_filtered)
    variance_x = np.var(x_filtered)
    variance_y = np.var(y_filtered)
    kurt_x = kurtosis(x_filtered)
    kurt_y = kurtosis(y_filtered)
    skew_x = skew(x_filtered)
    skew_y = skew(y_filtered)

    # Obliczanie magnitudy FFT
    fft_x = np.abs(fft(x_filtered))
    fft_y = np.abs(fft(y_filtered))
    fft_magnitude = np.mean(fft_x) + np.mean(fft_y)

    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_x': std_x,
        'std_y': std_y,
        'max_x': max_x,
        'min_x': min_x,
        'max_y': max_y,
        'min_y': min_y,
        'fft_magnitude': fft_magnitude,
        'variance_x': variance_x,
        'variance_y': variance_y,
        'kurtosis_x': kurt_x,
        'kurtosis_y': kurt_y,
        'skewness_x': skew_x,
        'skewness_y': skew_y
    }

def is_point_realistic(part_name, keypoint, previous_keypoints, margin=50):
    """Sprawdza, czy punkt kluczowy jest realistyczny na podstawie jego pozycji i ruchu"""
    if not previous_keypoints or np.isnan(keypoint).any():
        return True  # Pierwsza klatka lub brak punktu, brak porównań

    previous_keypoint = previous_keypoints[-1]  # Ostatni punkt kluczowy z historii

    distance = np.linalg.norm(np.array(keypoint) - np.array(previous_keypoint))  # Odległość między punktami
    max_distance_threshold = 100  # Maksymalna dopuszczalna odległość między klatkami

    if distance > max_distance_threshold or keypoint[0] <= margin or keypoint[1] <= margin:
        return False  # Punkt nie jest realistyczny, jeśli przekracza próg odległości

    return True

def is_point_stable(part_name, keypoint, history, threshold=STABILITY_THRESHOLD, margin=STABILITY_MARGIN):
    """Sprawdza, czy punkt kluczowy jest stabilny przez co najmniej `threshold` klatek"""
    if part_name not in history:
        history[part_name] = deque(maxlen=threshold)  # Inicjalizacja historii dla danego punktu

    history[part_name].append(keypoint)  # Dodanie aktualnego punktu do historii

    if len(history[part_name]) < threshold:
        return False  # Nie wystarczająca liczba klatek, by ocenić stabilność

    positions = np.array(history[part_name])
    x_mean = np.mean(positions[:, 0])
    y_mean = np.mean(positions[:, 1])

    # Obliczanie odchyleń od średniej
    deviations = np.sqrt((positions[:, 0] - x_mean) ** 2 + (positions[:, 1] - y_mean) ** 2)
    return np.all(deviations <= margin)  # Sprawdzenie, czy wszystkie odchylenia są w granicach marginesu

def predict_and_save_pose(video_path, margin=50):
    """Funkcja do przewidywania postawy z wideo i zapisywania wyników"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Nie można otworzyć pliku wideo.")  # Sprawdzenie, czy plik wideo został poprawnie otwarty

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inicjalizacja zapisu wideo
    out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Inicjalizacja słowników do przechowywania danych o częściach ciała
    body_parts_over_time = {part: [] for part in body_parts_mapping.keys()}
    body_parts_stability_history = {}

    # Otwieranie pliku do zapisu punktów kluczowych
    with open('points.txt', 'w') as points_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(input_image)

            frame_keypoints = {}

            if results.pose_landmarks:
                image_height, image_width, _ = frame.shape
                predicted_keypoints = [[landmark.x * image_width, landmark.y * image_height] for landmark in results.pose_landmarks.landmark]

                # Obliczanie pozycji szyi
                neck_position = calculate_neck(predicted_keypoints)
                predicted_keypoints.append(neck_position)

                # Aktualizacja mapowania body_parts_mapping, aby używać neck_position
                body_parts_mapping["neck"] = [len(predicted_keypoints) - 1]

                for part_name, part_indices in body_parts_mapping.items():
                    part_keypoints = []
                    if len(part_indices) > 1:
                        # Obliczanie średniej pozycji dla grupy punktów kluczowych
                        x_coords = [predicted_keypoints[i][0] for i in part_indices if i < len(predicted_keypoints)]
                        y_coords = [predicted_keypoints[i][1] for i in part_indices if i < len(predicted_keypoints)]
                        x_mean = np.mean(x_coords)
                        y_mean = np.mean(y_coords)
                        part_keypoints.append([x_mean, y_mean])
                    else:
                        # Sprawdzenie, czy punkt kluczowy jest realistyczny i stabilny
                        for i in part_indices:
                            if i < len(predicted_keypoints):
                                x, y = predicted_keypoints[i]
                                if not is_point_realistic(part_name, [x, y], body_parts_over_time[part_name], margin):
                                    x, y = np.nan, np.nan
                                if not is_point_stable(part_name, [x, y], body_parts_stability_history):
                                    x, y = np.nan, np.nan
                                part_keypoints.append([x, y])
                            else:
                                part_keypoints.append([np.nan, np.nan])

                    body_parts_over_time[part_name].extend(part_keypoints)
                    frame_keypoints[part_name] = part_keypoints

                # Rysowanie punktów kluczowych na obrazie
                for kp in predicted_keypoints:
                    if not (kp[0] <= margin or kp[1] <= margin or kp[0] >= (image_width - margin) or kp[1] >= (image_height - margin)):
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 20, (0, 255, 255), -1)

            else:
                # Jeśli nie wykryto punktów kluczowych, dodaj NaN
                for part_name in body_parts_mapping.keys():
                    body_parts_over_time[part_name].append([np.nan, np.nan])
                    frame_keypoints[part_name] = [[np.nan, np.nan]]

            # Zapis punktów kluczowych do pliku
            points_file.write(json.dumps(frame_keypoints) + "\n")
            # Zapis aktualnej klatki do wideo
            out_video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()  # Zamknięcie pliku wideo
    out_video.release()  # Zamknięcie pliku wideo wyjściowego

    # Obliczanie statystyk dla wszystkich zebranych danych
    body_parts_statistics = {}
    for part_name, keypoints in body_parts_over_time.items():
        stats = get_statistics(keypoints)
        body_parts_statistics[part_name] = stats

    # Zapis statystyk do pliku JSON
    with open('predicted_keypoints.txt', 'w') as f:
        json.dump(body_parts_statistics, f, indent=4)

    return 'output_video.mp4', body_parts_statistics
