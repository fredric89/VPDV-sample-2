import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pitch_utils import bandpass_filter, autocorrelation_pitch
from pydub import AudioSegment
import io

st.set_page_config(page_title="Voice Pitch Detector", layout="centered")

st.title("🎤 Voice Pitch Detection and Visualization")
st.write("Upload a voice recording (`.wav` or `.mp3`) to analyze and visualize pitch over time.")

audio_file = st.file_uploader("Upload a `.wav` or `.mp3` file", type=["wav", "mp3"])

if audio_file is not None:
    # Detect file type
    file_bytes = audio_file.read()

    try:
        if audio_file.name.endswith(".mp3"):
            # Convert MP3 to WAV in memory
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="mp3")
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            y, sr = librosa.load(wav_io, sr=None, mono=True)
        else:
            # If already WAV
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    except Exception as e:
        st.error(f"Audio loading failed: {e}")
        st.stop()

    st.audio(file_bytes, format="audio/wav" if audio_file.name.endswith(".wav") else "audio/mp3")
    st.success(f"Audio loaded! Duration: {len(y)/sr:.2f} seconds, Sample Rate: {sr} Hz")

    # Pre-processing
    st.subheader("📦 Pre-processing Audio")
    st.text("Applying bandpass filter (80–300 Hz)...")
    y_filtered = bandpass_filter(y, sr)

    # Pitch Detection
    st.subheader("📈 Pitch Detection in Progress...")
    times, pitches = autocorrelation_pitch(y_filtered, sr)

    # Visualization
    st.subheader("🔍 Pitch Over Time")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, pitches, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title("Pitch Curve")
    ax.grid(True)
    st.pyplot(fig)

    st.success("Pitch detection and visualization complete!")
else:
    st.info("Please upload a `.wav` or `.mp3` file to begin.")
