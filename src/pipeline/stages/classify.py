from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.inference.base import RunInference
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

# from src.pipeline.config import load_pipeline_config
# config = load_pipeline_config()

import psutil

def print_available_ram():
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert from bytes to GB
    total_ram = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Total RAM: {total_ram:.2f} GB")

print_available_ram()


class BaseClassifier(beam.PTransform): 
    name = "BaseClassifier"

    def __init__(self, config):
        self.config = config
        self.source_sample_rate = config.audio.source_sample_rate
        self.model_sample_rate = config.classify.model_sample_rate
        self.batch_duration = config.classify.batch_duration
        
    def _preprocess(self, pcoll):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        logging.info(f"Preprocessing signal for {key} ...")

        # Ensure signal is a numpy array
        if type(signal) == list:
            signal = np.array(signal, dtype=np.float32)  # TODO fix inline conversion
        elif type(signal) == np.ndarray:
            pass
        else:
            raise ValueError(f"Invalid signal type: {type(signal)}")
        
        # Resample
        signal = self._resample(signal)

        batch_samples = self.batch_duration * self.source_sample_rate

        if signal.shape[0] > batch_samples:
            logging.info(f"Signal size exceeds max sample size {batch_samples}.")

            split_indices = [batch_samples*(i+1) for i  in range(math.floor(signal.shape[0] / batch_samples))]
            signal_batches = np.array_split(signal, split_indices)
            logging.info(f"Split signal into {len(signal_batches)} batches of size {batch_samples}.")
            logging.info(f"Size of final batch {len(signal_batches[1])}")

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
        logging.info(f"Postprocessing signal ...")
        return pcoll
    
    def _get_model(self):
        model = hub.load(self.model_path)
        logging.info(f"Model loaded from {self.model_path}")
        logging.debug(f"Model score: {model.signatures['score']}")
        return model
    
        # def _get_model_tf_v1(self):
        #     import tensorflow.compat.v1 as tf

        #     FILENAME = 'gs://bioacoustics-www1/sounds/Cross_02_060203_071428.d20_7.wav'

        #     graph = tf.Graph()
        #     with graph.as_default():
        #         model = hub.load('https://kaggle.com/models/google/humpback-whale/frameworks/TensorFlow2/variations/humpback-whale/versions/1')

        #         filename = tf.placeholder(tf.string)
        #         waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(filename))

        #         waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        #         context_step_samples = tf.cast(sample_rate, tf.int64)
        #         score_fn = model.signatures['score']
        #         scores = score_fn(
        #             waveform=waveform, context_step_samples=context_step_samples
        #         )
    
    def _resample(self, signal):
        logging.info(f"Resampling signal from {self.source_sample_rate} to {self.model_sample_rate}")

        return librosa.resample(
            signal, 
            orig_sr=self.source_sample_rate, 
            target_sr=self.model_sample_rate
        )    


class HumpbackWhaleClassifier(BaseClassifier):
    """
    Model docs: https://tfhub.dev/google/humpback_whale/1
    """
    name = "HumpbackWhaleClassifier"

    def __init__(self, config):
        super().__init__(config)
        self.model_path = config.classify.model_url
        self.hydrophone_sensitivity = config.classify.hydrophone_sensitivity
    
    def expand(self, pcoll):
        return (
            pcoll
            | "Preprocess"  >> beam.ParDo(self._preprocess)
            # | "Batch"       >> beam.BatchElements(min_batch_size=1, max_batch_size=2)
            | "Classify"    >> beam.ParDo(GoogleHumpbackWhaleInferenceDoFn(self.config))
            # | "Postprocess" >> beam.Map(self._postprocess)
        )
    
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

        plot_path = self.config.classify.plot_path_template.format(
            year=start.year,
            month=start.month,
            day=start.day,
            plot_name=key
        )
        plt.savefig(plot_path)
        plt.show()


model = hub.load("https://tfhub.dev/google/humpback_whale/1")
class GoogleHumpbackWhaleInferenceDoFn(beam.DoFn):
    name = "GoogleHumpbackWhaleInferenceDoFn"

    def __init__(self, config):
        self.session = None
        self.config = config
        self.model_path = config.classify.model_url
        self.model_sample_rate = config.classify.model_sample_rate
        
    def setup(self):
        # Load the model once per worker
        self.model = model
        self.score_fn = self.model.signatures["score"]
        logging.info(f"Model loaded inside {self.name}")
    
    def process(self, element):
        key, signal = element
        # key = "test element"
        # signal = element[0][0]
        # logging.info(f"Classifying signal for {key} ...")
        logging.info(f"   signal.shape = {len(signal)}")
        logging.info(f"   signal.dtype = {type(signal)}")

        # if len(signal) == 0:
        #     logging.info(f"Empty signal for {key}")
        #     return (key, None)

        # Reshape signal
        logging.info(f"   inital input: len(signal) = {len(signal)}")
        signal = np.array(signal, dtype=np.float32)
        waveform1 = np.expand_dims(signal, axis=1)
        waveform_exp = tf.expand_dims(waveform1, 0)
        logging.info(f"   final input: waveform_exp.shape = {waveform_exp.shape}")
        logging.info(f"   final input: waveform_exp.dtype = {waveform_exp.dtype}")
        logging.info(f"   final input: type(waveform_exp) = {type(waveform_exp)}")


        # Inference
        if waveform_exp.shape[1] < self.model_sample_rate:
            "skip short signal"
            return (key, None)
        
        print("before inference")
        print_available_ram()

        # results = self.score_fn(
        #     waveform=waveform_exp,
        #     context_step_samples=int(self.model_sample_rate)
        # )["scores"]
        results = self.model.score(
            waveform=waveform_exp,
            context_step_samples=int(self.model_sample_rate)
        )["scores"]

        print("after inference")
        print_available_ram()

        # results = waveform_exp
        logging.info(f"   results.shape = {results.shape}")
        
        yield (key, results)
