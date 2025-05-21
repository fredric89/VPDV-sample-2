import numpy as np
import scipy.signal

def bandpass_filter(signal, sr, low=80, high=300):
    sos = scipy.signal.butter(10, [low, high], btype='band', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, signal)

def autocorrelation_pitch(y, sr, frame_size=2048, hop_size=512):
    pitches = []
    times = []
    for i in range(0, len(y) - frame_size, hop_size):
        frame = y[i:i + frame_size]
        frame = frame - np.mean(frame)  # remove DC
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        d = np.diff(corr)
        try:
            start = np.where(d > 0)[0][0]
            peak = np.argmax(corr[start:]) + start
            pitch = sr / peak if peak > 0 else 0
        except:
            pitch = 0
        pitches.append(pitch)
        times.append(i / sr)
    return times, pitches
