import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model(r"C:\Users\shubh\OneDrive\Desktop\dsp_cry_corpus\dsp_cry_corpus_final.h5")

def preprocess_audio(audio_file):
    stft_db = []
    signal, sr = librosa.load(audio_file, duration=5.0)
    stft_matrix = librosa.stft(signal, n_fft=2046, hop_length=2046)
    stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=1)
    stft_db = stft_db.reshape(128, 432, 1)
    return stft_db

def predict_label(audio_file):
    label = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired'] 
    labels = np.array(label) 
    processed_audio = preprocess_audio(audio_file)
    predicted_prob = model.predict(np.array([processed_audio]))
    predicted_label_index = np.argmax(predicted_prob)
    predicted_label = labels[predicted_label_index]
    return predicted_label


def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_label = predict_label(file_path)
        result_label.config(text="Predicted label: " + predicted_label)

# Create the main window
root = tk.Tk()
root.title("Audio Label Prediction")
root.geometry("800x600")
# Create browse button
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=10)

# Create label to display result
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the main event loop
root.mainloop()
