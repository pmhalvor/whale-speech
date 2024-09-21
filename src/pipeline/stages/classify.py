from apache_beam.io import filesystems
from datetime import datetime

import apache_beam as beam
import io
import librosa
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import time
import tensorflow_hub as hub
import tensorflow as tf

from config import load_pipeline_config


config = load_pipeline_config()


class BaseClassifier(beam.PTransform): 
    name = "BaseClassifier"


    def __init__(self):
        self.source_sample_rate = config.audio.source_sample_rate
        self.model_sample_rate = config.classify.model_sample_rate
        
        self.model_path = config.classify.model_path

    def _preprocess(self, pcoll):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        # Resample
        signal = self._resample(signal)

        batch_samples = self.batch_duration * self.sample_rate

        if signal.shape[0] > batch_samples:
            logging.debug(f"Signal size exceeds max sample size {batch_samples}.")

            split_indices = [batch_samples*(i+1) for i  in range(math.floor(signal.shape[0] / batch_samples))]
            signal_batches = np.array_split(signal, split_indices)
            logging.debug(f"Split signal into {len(signal_batches)} batches of size {batch_samples}.")
            logging.debug(f"Size fo final batch {len(signal_batches[1])}")

            for batch in signal_batches:
                yield (key, batch)
        else:
            yield (key, signal)
    
    def _build_key(
            self,
            start_time: datetime,
            end_time: datetime,
            encounter_ids: list,
        ):
        start_str = start_time.strftime('%Y%m%dT%H%M%S')
        end_str = end_time.strftime('%H%M%S')
        encounter_str = "_".join(encounter_ids)
        return f"{start_str}-{end_str}_{encounter_str}"

    def _postprocess(self, pcoll):
        return pcoll
    
    def _get_model(self):
        model = hub.load(self.model_path)
        return model
    
    def _resample(self, signal):
        logging.info(
            f"Resampling signal from {self.source_sample_rate} to {self.model_sample_rate}")
        return librosa.resample(
            signal, 
            orig_sr=self.source_sample_rate, 
            target_sr=self.model_sample_rate
        )    


class GoogleHumpbackWhaleClassifier(BaseClassifier):
    """
    Model docs: https://tfhub.dev/google/humpback_whale/1
    """
    name = "GoogleHumpbackWhaleClassifier"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self._get_model()
        self.score_fn = self.model.signatures['score']
        self.metadata_fn = self.model.signatures['metadata']

    
    def expand(self, pcoll):
        return (
            pcoll
            | "Preprocess"  >> beam.Map(self._preprocess)
            | "Classify"    >> beam.Map(self._classify)
            | "Postprocess" >> beam.Map(self._postprocess)
        )
    

    def _classify(self, pcoll, ):
        key, signal = pcoll
    
        start_classify = time.time()

        # We specify a 1-sec score resolution:
        context_step_samples = tf.cast(self.model_sample_rate, tf.int64)

        logging.info(f'\n==> Applying model ...')
        logging.debug(f'   inital input: len(signal_10kHz) = {len(signal)}')

        waveform1 = np.expand_dims(signal, axis=1)
        waveform_exp = tf.expand_dims(waveform1, 0)  # makes a batch of size 1
        logging.debug(f"   final input: waveform_exp.shape = {waveform_exp.shape}")

        signal_scores = self.score_fn(
            waveform=waveform_exp,
            context_step_samples=context_step_samples
        )
        score_values = signal_scores['scores'].numpy()[0]
        logging.info(f'==> Model applied.  Obtained {len(score_values)} score_values')
        logging.info(f'==> Elapsed time: {time.time() - start_classify} seconds')

        return (key, score_values)
    
    def _plot_spectrogram_scipy(self, signal, epsilon = 1e-15):
        # Compute spectrogram:
        w = scipy.signal.get_window('hann', self.sample_rate)
        f, t, psd = scipy.signal.spectrogram(
            signal,  # TODO make sure this is resampled signal
            self.model_sample_rate,
            nperseg=self.model_sample_rate,
            noverlap=0,
            window=w,
            nfft=self.model_sample_rate,
        )
        psd = 10*np.log10(psd+epsilon) - self.hydrophone_sensitivity

        # Plot spectrogram:
        fig = plt.figure(figsize=(20, round(20/3))) # 3:1 aspect ratio
        plt.imshow(
            psd,
            aspect='auto',
            origin='lower',
            vmin=30,
            vmax=90,
            cmap='Blues',
        )
        plt.yscale('log')
        y_max = self.model_sample_rate / 2
        plt.ylim(10, y_max)

        plt.colorbar()

        plt.xlabel('Seconds')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Calibrated spectrum levels, 16 {self.sample_rate / 1000.0} kHz data')


    def _plot_scores(self, pcoll, scores, med_filt_size=None):
        audio, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        # repeat last value to also see a step at the end:
        scores = np.concatenate((scores, scores[-1:]))
        x = range(len(scores))
        plt.step(x, scores, where='post')
        plt.plot(x, scores, 'o', color='lightgrey', markersize=9)

        plt.grid(axis='x', color='0.95')
        plt.xlim(xmin=0, xmax=len(scores) - 1)
        plt.ylabel('Model Score')
        plt.xlabel('Seconds')

        if med_filt_size is not None:
            scores_int = [int(s[0]*1000) for s in scores]
            meds_int = scipy.signal.medfilt(scores_int, kernel_size=med_filt_size)
            meds = [m/1000. for m in meds_int]
            plt.plot(x, meds, 'p', color='black', markersize=9)

        plot_path = config.classify.plot_path_template.format(
            year=start.year,
            month=start.month,
            day=start.day,
            plot_name=key
        )
        plt.savefig(plot_path)
        plt.show()
