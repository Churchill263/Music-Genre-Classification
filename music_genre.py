# Import libraries
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
import pickle

# Load the saved model
model = load_model("audio_classification_19_02_04.hdf5")

with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)


# Define the Streamlit application
st.title("Audio Genre Classification")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Extract features from the uploaded file
    audio, sample_rate = librosa.load(uploaded_file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Predict genre
    pred = model.predict(np.expand_dims(mfccs_scaled_features, axis=0))
    predicted_class = encoder.inverse_transform(np.argmax(pred, axis=1))[0]

    # Display prediction
    st.write(f"Predicted Genre: {predicted_class}")
else:
    st.write("Please upload an audio file.")
