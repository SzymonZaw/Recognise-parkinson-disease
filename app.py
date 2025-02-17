import pandas as pd
import numpy as np
import torch
import streamlit as st
from main import parkinsonTrain  # Importuje funkcję trenowania modelu z pliku main.py
from poseEst import predict_and_save_pose  # Importuje funkcję estymacji pozycji z pliku poseEst.py

def main():
    # Definiowanie stylów CSS dla aplikacji
    st.markdown("""
        <style>
            .reportview-container {
                background: #f0f2f6;
            }
            .sidebar .sidebar-content {
                background: #f0f2f6;
            }
            .header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: #0072C6;
                padding: 20px;
                color: white;
            }
            .header img {
                border-radius: 50%;
                width: 100px;
                height: 100px;
                margin-right: 20px;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 0;
                font-size: 1.2em;
            }
            .content {
                padding: 20px;
            }
            .content h2 {
                color: #0072C6;
            }
            .content p, .content ul {
                font-size: 1.1em;
                line-height: 1.6;
            }
            .footer {
                text-align: center;
                padding: 10px;
                font-size: 0.9em;
                color: #555;
                border-top: 1px solid #ddd;
                margin-top: 20px;
            }
            .footer a {
                color: #0072C6;
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
        </style>
    """, unsafe_allow_html=True)  # Użycie CSS do stylizacji aplikacji

    # Opcja wyboru języka w bocznym pasku
    language = st.sidebar.selectbox('Choose Language', ['Polski', 'English'])

    # Zdefiniowanie etykiet i treści w zależności od wybranego języka
    if language == 'Polski':  # Polski język interfejsu
        nav_labels = ('Strona Główna', 'Trenowanie modelu i predykcja', 'Estymacja pozycji i zapis', 'Odnośniki')
        home_title = "Strona Główna"
        home_description = """
            <h2>Tytuł Pracy:</h2>
            <p><strong>Rozpoznawanie stopnia zaawansowania choroby Parkinsona na podstawie analiz nagrań wideo</strong></p>
            <p>Praca dotyczy opracowania modelu uczenia maszynowego do rozpoznawania stopnia zaawansowania choroby Parkinsona na podstawie analizy nagrań wideo.</p>
            <p>W ramach projektu wymagane jest:</p>
            <ul>
                <li>Przegląd literatury dotyczącej przebiegu choroby Parkinsona oraz modeli uczenia maszynowego stosowanych w diagnozowaniu tej choroby.</li>
                <li>Przetwarzanie nagrań wideo pacjentów, w tym analiza klatek wideo i ekstrakcja cech.</li>
                <li>Opracowanie modeli rozpoznawania przestrzennego ciała i klasyfikacji stopnia zaawansowania choroby Parkinsona.</li>
                <li>Trenowanie modeli i integracja ich w aplikację do automatycznej analizy wideo.</li>
                <li>Testowanie i ocena skuteczności modeli.</li>
            </ul>
            <p>Użyj paska bocznego, aby nawigować między różnymi funkcjami aplikacji.</p>
        """
        model_training_label = "### Przewidywanie stopnia choroby Parkinsona. Możesz przesłać plik utworzony po przewidzeniu punktów ciała z filmu z zakładki 'Estymacja pozycji i zapis' o nazwie predicted_keypoints.txt:"
        pose_estimation_label = "### Przewiduj współrzędne pozycji osoby w wideo. Wyświetla wideo z punktami kluczowymi i zapisuje przewidziane punkty kluczowe do pliku."
        references_title = "Odnośniki:"
        references_links = """
            <ul>
                <li><a href="https://docs.streamlit.io/" target="_blank">Dokumentacja Streamlit</a></li>
                <li><a href="https://github.com/google/mediapipe" target="_blank">GitHub Mediapipe</a></li>
                <li><a href="https://pytorch.org/docs/stable/index.html" target="_blank">Dokumentacja PyTorch</a></li>
                <li><a href="https://drive.google.com/drive/folders/1C5kc8K3Oh\_RpW6NsHXYNfP8eYLpJv5Je?usp=sharing" target="_blank">Dokumentacja i kod pracy dyplomowej</a></li>
            </ul>
        """
        about_info = "Ta aplikacja umożliwia przewidywanie przestrzennych punktów ciała na podstawie filmów oraz określanie stopnia zaawansowania choroby Parkinsona."

    else:  # English language interface
        nav_labels = ('Home', 'Model Training and Prediction', 'Pose Estimation', 'References')
        home_title = "Home"
        home_description = """
            <h2>Thesis Title:</h2>
            <p><strong>Recognizing the severity of Parkinson's disease based on analyzes of video recordings</strong></p>
            <p>This project focuses on developing a machine learning model to recognize the progression of Parkinson's disease based on video analysis.</p>
            <p>The project involves:</p>
            <ul>
                <li>Reviewing literature on Parkinson's disease and the application of machine learning models for diagnosis.</li>
                <li>Processing patient video data, including frame analysis and feature extraction.</li>
                <li>Developing models for spatial body recognition and classification of Parkinson's disease progression.</li>
                <li>Training the models and integrating them into an application for automatic video analysis.</li>
                <li>Testing and evaluating the effectiveness of the models.</li>
            </ul>
            <p>Use the sidebar to navigate between different functionalities of the application.</p>
        """
        model_training_label = "### Upload the predicted_keypoints.txt file:"
        pose_estimation_label = "### Predict the pose coordinates of a person in a video. Displays the video with keypoints and saves predicted keypoints to a file."
        references_title = "References:"
        references_links = """
            <ul>
                <li><a href="https://docs.streamlit.io/" target="_blank">Streamlit Documentation</a></li>
                <li><a href="https://github.com/google/mediapipe" target="_blank">Mediapipe GitHub</a></li>
                <li><a href="https://pytorch.org/docs/stable/index.html" target="_blank">PyTorch Documentation</a></li>
                <li><a href="https://www.overleaf.com/read/hqqtgbgfyqnd#699053" target="_blank">Thesis Documentation</a></li>
            </ul>
        """
        about_info = "This application enables the prediction of spatial body points based on videos and the determination of the severity of Parkinson's disease."

    # Boczne menu nawigacyjne
    st.sidebar.title("Navigation" if language == 'English' else "Nawigacja")
    selected_app = st.sidebar.selectbox('Choose an application' if language == 'English' else 'Wybierz aplikację', nav_labels)

    # Zależnie od wybranej opcji z menu nawigacyjnego, wyświetl odpowiednią stronę
    if selected_app == 'Strona Główna' or selected_app == 'Home':
        header_description = 'Rozpoznawanie stopnia zaawansowania choroby Parkinsona' if language == 'Polski' else "Recognizing the severity of Parkinson's disease"
        # Wyświetlenie nagłówka strony głównej
        st.markdown(f"""
            <div class="header">
                <div>
                    <h1>{home_title}</h1>
                    <p>{header_description}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Wyświetlenie opisu strony głównej
        st.markdown(f"""
            <div class="content">
                {home_description}
            </div>
        """, unsafe_allow_html=True)

    elif selected_app == 'Trenowanie modelu i predykcja' or selected_app == 'Model Training and Prediction':
        st.markdown(f"## {'Trenowanie modelu i predykcja' if language == 'Polski' else 'Model Training and Prediction'}")
        st.markdown(model_training_label)

        # Wyniki na skali UDysRS
        if language == 'Polski':
            st.markdown("""
                ### Wyniki na skali UDysRS:
                - **0**: Brak objawów. Osoba nie wykazuje żadnych oznak dyskinezji.
                - **1**: Łagodne objawy. Objawy dyskinezji są zauważalne, ale nie wpływają znacząco na codzienne funkcjonowanie.
                - **2**: Umiarkowane objawy. Dyskinezy są wyraźne i zaczynają wpływać na codzienne życie, np. utrudniając niektóre czynności.
                - **3**: Ciężkie objawy. Dyskinezy są nasilone, znacząco zaburzając funkcjonowanie, co może prowadzić do dużych trudności w poruszaniu się i wykonywaniu codziennych zadań.
            """)
        else:
            st.markdown("""
                ### UDysRS Scale Results:
                - **0**: No symptoms. The person shows no signs of dyskinesia.
                - **1**: Mild symptoms. Dyskinesia symptoms are noticeable but do not significantly impact daily functioning.
                - **2**: Moderate symptoms. Dyskinesias are apparent and begin to affect daily life, such as hindering some tasks.
                - **3**: Severe symptoms. Dyskinesias are intense, significantly disrupting functioning, which may lead to major difficulties in movement and performing daily tasks.
            """)

        # Uploader pliku tekstowego do trenowania modelu
        uploaded_file = st.file_uploader("Upload File" if language == 'English' else "Prześlij plik", type=["txt"])

        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")  # Odczytanie zawartości pliku
            parkinsonTrain(file_content)  # Uruchomienie trenowania modelu na podstawie zawartości pliku
            st.success("File successfully uploaded and processed." if language == 'English' else "Plik został pomyślnie przesłany i przetworzony.")
        else:
            st.warning("Please upload a file to continue." if language == 'English' else "Proszę przesłać plik, aby kontynuować.")

    elif selected_app == 'Estymacja pozycji i zapis' or selected_app == 'Pose Estimation':
        st.markdown(f"## {'Estymacja pozycji i zapis' if language == 'Polski' else 'Pose Estimation and Save'}")
        st.markdown(pose_estimation_label)

        # Uploader pliku wideo do analizy
        uploaded_file = st.file_uploader("Choose a video file" if language == 'English' else "Wybierz plik wideo", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            video_path = "temp_video.mp4"
            # Zapisanie przesłanego wideo do pliku tymczasowego
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(video_path)  # Wyświetlenie wideo w aplikacji
            if st.button('Predict and Save Pose' if language == 'English' else 'Przewiduj i Zapisz Pozycję'):
                output_video, keypoints = predict_and_save_pose(video_path)  # Przeprowadzenie estymacji pozycji
                st.success('Pose estimation completed and video saved!' if language == 'English' else 'Estymacja pozycji zakończona, wideo zapisane!')

                # Przyciski do pobierania zapisanego wideo
                st.download_button(label="Download Video" if language == 'English' else "Pobierz wideo", data=open(output_video, 'rb').read(), file_name='output_video.mp4')

                # Przyciski do pobierania predicted_keypoints.txt
                with open("predicted_keypoints.txt", 'rb') as f:
                    st.download_button(
                        label="Download Keypoints File" if language == 'English' else "Pobierz plik z punktami kluczowymi",
                        data=f.read(),
                        file_name='predicted_keypoints.txt'
                    )
        else:
            st.warning("Please upload a video file to continue." if language == 'English' else "Proszę przesłać plik wideo, aby kontynuować.")

    elif selected_app == 'Odnośniki' or selected_app == 'References':
        # Wyświetlenie odnośników do dokumentacji i innych źródeł
        st.markdown(f"""
            <div class="content">
                {references_title}
                {references_links}
            </div>
        """, unsafe_allow_html=True)

    # Informacje o aplikacji w pasku bocznym
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About" if language == 'English' else "### O aplikacji")
    st.sidebar.info(about_info)

    # Stopka aplikacji
    st.markdown("""
        <div class="footer">
            <p>&copy; 2024 Szymon Zawadzki. All rights reserved.</p>
            <p>Contact: <a href="mailto:szyzaw000@pbs.edu.pl">szyzaw000@pbs.edu.pl</a></p>
            <p>Follow me on <a href="https://github.com/SzymonZaw" target="_blank">GitHub</a></p>
        </div>
    """, unsafe_allow_html=True)

# Wywołanie funkcji main() gdy uruchamiamy skrypt
if __name__ == '__main__':
    main()
