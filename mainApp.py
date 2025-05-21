import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pitch_utils import bandpass_filter, autocorrelation_pitch

st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

st.title("🎤 Voice Pitch Detection and Visualization")
st.write("Upload a voice recording (`.wav` or `.mp3`) to analyze and visualize pitch over time.")

# 🔄 1. Allow .mp3 and .wav files
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    # 🔄 2. Load using librosa (supports mp3 and wav)
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        st.stop()

    # Optional: Listen to uploaded file
    st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")

    st.success(f"Audio loaded! Duration: {len(y)/sr:.2f} seconds, Sample Rate: {sr} Hz")

    # Pre-processing
    st.subheader("📦 Pre-processing Audio")
    st.text("Converting to mono and applying bandpass filter...")
    y_filtered = bandpass_filter(y, sr)

    # Pitch Detection
    st.subheader("📈 Pitch Detection in Progress...")
    times, pitches = autocorrelation_pitch(y_filtered, sr)

    # Visualization
    st.subheader("🔍 Pitch Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, pitches, label="Pitch (Hz)", color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title("Pitch Curve")
    ax.grid(True)
    st.pyplot(fig)

    st.success("Pitch detection and visualization complete!")

else:
    st.info("Please upload a `.wav` or `.mp3` file to begin.")
