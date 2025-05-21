import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pitch_utils import bandpass_filter, autocorrelation_pitch
import ffmpeg
import io
import soundfile as sf

st.set_page_config(page_title="Voice Pitch Detector", layout="centered")
st.title("🎤 Voice Pitch Detection and Visualization")
st.write("Upload a `.wav` or `.mp3` file to analyze and visualize pitch over time.")

def load_audio_ffmpeg(file, format):
    """Decode MP3/WAV using ffmpeg-python and return raw waveform + sr"""
    try:
        out, _ = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format='wav', acodec='pcm_s16le', ac=1, ar='44100')
            .run(input=file.read(), capture_stdout=True, capture_stderr=True)
        )
        audio_np, sr = sf.read(io.BytesIO(out))
        return audio_np, sr
    except Exception as e:
        st.error(f"ffmpeg decoding failed: {e}")
        return None, None

audio_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if audio_file is not None:
    file_format = audio_file.name.split('.')[-1].lower()
    y, sr = load_audio_ffmpeg(audio_file, file_format)

    if y is None:
        st.stop()

    st.audio(audio_file, format=f'audio/{file_format}')
    st.success(f"Audio loaded! Duration: {len(y)/sr:.2f} seconds, Sample Rate: {sr} Hz")

    st.subheader("📦 Pre-processing Audio")
    st.text("Filtering signal to human speech frequency range...")
    y_filtered = bandpass_filter(y, sr)

    st.subheader("📈 Detecting Pitch...")
    times, pitches = autocorrelation_pitch(y_filtered, sr)

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
    st.info("Please upload a `.wav` or `.mp3` file.")
