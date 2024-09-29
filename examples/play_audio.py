# NOTE simpleaudio is excluded from requirements due to non-portability.
import numpy as np
import simpleaudio as sa

# Load the audio file
audio_file = "data/audio/butterworth/2016/12/20161221T004930-005030-9182.npy"
audio_array = np.load(audio_file)

# Normalize audio to 16-bit PCM format
audio = np.int16(audio_array/np.max(np.abs(audio_array)) * 32767)

# Play the audio using simpleaudio
play_obj = sa.play_buffer(audio, 1, 2, 16000)  # Channels=1, Bytes per sample=2, Sample rate=44100Hz

# Wait for playback to finish before exiting
play_obj.wait_done()