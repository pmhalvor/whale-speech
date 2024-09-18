from apache_beam.io import filesystems
from datetime import timedelta, datetime
from scipy.signal import butter, lfilter, find_peaks, sosfilt

import apache_beam as beam
import io
import logging
import numpy as np
import pandas as pd
import soundfile as sf

from config import load_pipeline_config


config = load_pipeline_config()


class Detection: # base detection transform 

    def parse_signal(
        self,
        signal,
        start_second = 0,
        seconds = 30,
        sample_rate = config.audio.source_sample_rate,
    ):

        sample_size = sample_rate * seconds
        sample_start = start_second * sample_rate
        sample_stop = sample_start + sample_size

        sample_signal = signal[sample_start:sample_stop]

        return (
            sample_start,
            sample_stop,
            sample_signal
        )
    

    def seconds_to_flags(self, seconds, sample_rate):
        return np.array([int(s*sample_rate) for s in seconds])


class Butterworth(Detection):
    lowcut = 50
    highcut = 1500
    order = 5
    window_size = 512


    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # b, a = butter(order, [low, high], btype='band')
        sos = butter(order, [low, high], btype='band', output="sos")
        # return b, a
        return sos
    

    def frequency_based_detection(
            self,
            signal,
            sample_rate,
            lowcut=200,
            highcut=1500,
            window_size=512,
            threshold=0.015,
            order=5,
        ):
        # Apply bandpass filter
        # b, a = butter_bandpass(lowcut, highcut, sample_rate, order)
        sos = self.butter_bandpass(lowcut, highcut, sample_rate, order)
        # filtered_signal = lfilter(b, a, signal)
        filtered_signal = sosfilt(sos, signal)

        # Calculate energy in windows
        energy = np.array([
            sum(abs(filtered_signal[i:i+window_size]**2))
            for i in range(0, len(filtered_signal), window_size)
        ])

        # Normalize energy
        energy = energy / np.max(energy)

        # Find peaks above threshold
        peaks, _ = find_peaks(energy, height=threshold)

        # Convert peak indices to time
        peak_times = peaks * (window_size / sample_rate)

        return peak_times

