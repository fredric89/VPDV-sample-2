import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pitch_utils import bandpass_filter, autocorrelation_pitch

st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

st.title("🎤 Voice Pitch Detection and Visualization")
st.write("Upload a voice recording to analyze and visualize its pitch over time.")

# Upload .wav file
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if audio_file is not None:
    # Load audio
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    st.audio(audio_file, format="audio/wav")
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
    st.info("Please upload a WAV file (mono/stereo, any length).")
