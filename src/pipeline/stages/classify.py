import apache_beam as beam

from datetime import datetime
from typing import Dict, Any
from types import SimpleNamespace
from matplotlib import gridspec

import librosa
import logging
import numpy as np
import os 
import time

import requests
import math 
import matplotlib.pyplot as plt
import scipy.signal


logging.getLogger().setLevel(logging.INFO)


class BaseClassifier(beam.PTransform): 
    name = "BaseClassifier"


    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.source_sample_rate = config.audio.source_sample_rate

        self.batch_duration     = config.classify.batch_duration
        self.model_sample_rate  = config.classify.model_sample_rate
        self.model_url          = config.classify.model_url

        # plotting parameters
        self.hydrophone_sensitivity = config.classify.hydrophone_sensitivity
        self.med_filter_size        = config.classify.med_filter_size
        self.plot_scores            = config.classify.plot_scores
        self.plot_path_template     = config.classify.plot_path_template
        self.show_plots             = config.general.show_plots

    def _preprocess(self, pcoll):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        # Resample
        signal = self._resample(signal)
        logging.info(f"Resampled signal shape: {signal.shape}")

        # Expand final dimension
        signal = np.expand_dims(signal, axis=1)
        logging.info(f"Expanded signal shape: {signal.shape}")

        # Split signal into batches (if necessary)
        batch_samples = self.batch_duration * self.model_sample_rate

        if signal.shape[0] > batch_samples:
            logging.info(f"Signal size exceeds max sample size {batch_samples}.")

            split_indices = [batch_samples*(i+1) for i  in range(math.floor(signal.shape[0] / batch_samples))]
            signal_batches = np.array_split(signal, split_indices)
            logging.info(f"Split signal into {len(signal_batches)} batches of size {batch_samples}.")
            logging.info(f"Size fo final batch {len(signal_batches[1])}")

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
        # TODO: Refactor this to a common place accessible in all modules that use key
        start_str = start_time.strftime('%Y%m%dT%H%M%S')
        end_str = end_time.strftime('%H%M%S')
        encounter_str = "_".join(encounter_ids)
        return f"{start_str}-{end_str}_{encounter_str}"

    def _postprocess(self, pcoll):
        return pcoll
    
    def _resample(self, signal):
        logging.info(f"Resampling signal from {self.source_sample_rate} to {self.model_sample_rate}")
        return librosa.resample(
            signal, 
            orig_sr=self.source_sample_rate, 
            target_sr=self.model_sample_rate
        )    

    def _plot_scores(self, scores, t=None):
        # repeat last value to also see a step at the end:
        scores = np.concatenate((scores, scores[-1:]))
        x = range(len(scores))
        plt.step(x, scores, where='post')
        plt.plot(x, scores, 'o', color='lightgrey', markersize=9)

        plt.grid(axis='x', color='0.95')
        plt.xlim(xmin=0, xmax=len(scores) - 1)
        plt.ylabel('Model Score')
        plt.xlabel('Seconds')
        plt.xlim(0, len(t)) if t is not None else None

        if self.med_filter_size is not None:
            scores_int = [int(s[0]*1000) for s in scores]
            meds_int = scipy.signal.medfilt(scores_int, kernel_size=self.med_filter_size)
            meds = [m/1000. for m in meds_int]
            plt.plot(x, meds, 'p', color='black', markersize=9)

    def _plot_spectrogram_scipy(
            self,
            signal,
            epsilon = 1e-16,
        ):

        # Compute spectrogram:
        w = scipy.signal.get_window('hann', self.source_sample_rate)
        f, t, psd = scipy.signal.spectrogram(
            signal,
            self.source_sample_rate,
            nperseg=self.source_sample_rate,
            noverlap=0,
            window=w,
            nfft=self.source_sample_rate,
        )
        psd = 10*np.log10(psd+epsilon) - self.hydrophone_sensitivity
        logging.info(f"time: {t.shape}, freq: {f.shape}, psd: {psd.shape}")

        # Plot spectrogram:
        plt.imshow(
            psd,
            aspect='auto',
            origin='lower',
            vmin=30,
            vmax=90,
            cmap='Blues',
        )
        plt.yscale('log')
        y_max = self.source_sample_rate / 2
        plt.ylim(10, y_max)

        plt.xlabel('Seconds')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Calibrated spectrum levels, 16 {self.source_sample_rate / 1000.0} kHz data')
        return t, f, psd

    def _plot_audio(self, audio, key):
        plt.plot(audio)
        plt.xlabel('Samples')
        plt.xlim(0, len(audio))
        plt.ylabel('Energy')
        plt.title(f'Raw audio signal for {key}')

    def _plot(self, output):
        audio, start, end, encounter_ids, scores = output
        key = self._build_key(start, end, encounter_ids)

        if len(audio) == 0:
            logging.info("No audio to plot.")
            return
        
        fig = plt.figure(figsize=(24, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

        # Plot spectrogram:
        plt.subplot(gs[0])
        self._plot_audio(audio, key)

        # Plot spectrogram:
        plt.subplot(gs[1])
        t, _, _ = self._plot_spectrogram_scipy(audio)

        # Plot scores:
        fig.add_subplot(gs[2])
        self._plot_scores(scores, t=t)

        plt.tight_layout()

        plot_path = self.plot_path_template.format(
            year=start.year,
            month=start.month,
            day=start.day,
            plot_name=key
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # TODO refactor when running on GCP

        plt.savefig(plot_path)
        plt.show() if self.show_plots else plt.close()


class WhaleClassifier(BaseClassifier):
    def expand(self, pcoll):
        key_batch       = pcoll             | "Preprocess"              >> beam.ParDo(self._preprocess)
        batched_outputs = key_batch         | "Classify"                >> beam.ParDo(InferenceClient(self.config))
        grouped_outputs = batched_outputs   | "Combine batched_outputs" >> beam.CombinePerKey(ListCombine())
        outputs         = pcoll             | "Postprocess"             >> beam.Map(
            self._postprocess,
            grouped_outputs=beam.pvalue.AsDict(grouped_outputs), 
        )

        if self.plot_scores:
            outputs | "Plot scores" >> beam.Map(self._plot)

        return outputs
    
    def _postprocess(self, pcoll, grouped_outputs):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)
        scores = grouped_outputs.get(key, [])

        logging.info(f"Postprocessing {key} with signal {len(signal)} and scores {len(scores)}")

        return signal, start, end, encounter_ids, scores
    

class InferenceClient(beam.DoFn):
    def __init__(self, config: SimpleNamespace):

        self.model_url = config.classify.model_url
        self.retries = config.classify.inference_retries

    def process(self, element):
        key, batch = element

        # skip empty batches
        if len(batch) == 0:
            return {"key": key, "predictions": []}

        if isinstance(batch, np.ndarray):
            batch = batch.tolist()

        data = {
            "key": key,
            "batch": batch,
        }

        wait = 0 
        while wait < 5:
            try:
                response = requests.post(self.model_url, json=data)
                response.raise_for_status()
                break
            except requests.exceptions.ConnectionError as e:
                logging.info(f"Connection error: {e}")
                logging.info(f"Retrying in {wait*wait} seconds.")
                wait += 1
                time.sleep(wait*wait)

        response = requests.post(self.model_url, json=data)
        response.raise_for_status()

        predictions = response.json().get("predictions", [])

        logging.info(f"Received response:\n  predictions:{predictions}\n  key: {key}")

        yield (key, predictions)


class ListCombine(beam.CombineFn):
    name = "ListCombine"
    # TODO refactor this to a place accessible in both sift.py and here

    def create_accumulator(self):
        return []

    def add_input(self, accumulator, input):
        """
        Key is not available in this method, 
        though inputs are only added to accumulator under correct key.
        """
        logging.debug(f"Adding input {input} to {self.name} accumulator.")
        if isinstance(input, np.ndarray):
            input = input.tolist()
        accumulator += input
        return accumulator

    def merge_accumulators(self, accumulators):
        return [item for sublist in accumulators for item in sublist]

    def extract_output(self, accumulator):
        return accumulator
    

class WriteResults(beam.DoFn):
    def process(self, element):
        # TODO refactor to properly handle typing
        array = element[0]
        start = element[1]
        end = element[2]
        encounter_ids = element[3]

        file_path = self._get_export_path(
            start,
            end,
            encounter_ids,
        )

        logging.info(f"Writing audio to {file_path}")
        logging.info(f"Audio shape: {array.shape}")
        
        yield self._save_audio(array, file_path)
        

    def _save_audio(self, audio:np.array, file_path:str):
        # Write the numpy array to the file as .npy format
        with beam.io.filesystems.FileSystems.create(file_path) as f:
            np.save(f, audio)  # Save the numpy array in .npy format

        return file_path


def sample_run():
    signal = np.load("data/audio/butterworth/2016/12/20161221T004930-005030-9182.npy")
    data = (
        signal, 
        datetime.strptime("2016-12-21T00:49:30", "%Y-%m-%dT%H:%M:%S"),
        datetime.strptime("2016-12-21T00:50:30", "%Y-%m-%dT%H:%M:%S"),
        ["9182"]
    )

    # simulate config (avoids local import)
    config = SimpleNamespace(
        general = SimpleNamespace(
            show_plots=True,
        ),
        audio = SimpleNamespace(source_sample_rate=16_000),
        classify = SimpleNamespace(
            batch_duration=30, # seconds
            hydrophone_sensitivity=-168.8,
            model_sample_rate=10_000,
            model_url="http://127.0.0.1:5000/predict",
            plot_scores=True,
            plot_path_template="data/plots/results/{year}/{month:02}/{plot_name}.png",
            med_filter_size=3,
        ),
    )



    with beam.Pipeline() as p:
        output = (
            p 
            | beam.Create([(data)])
            | WhaleClassifier(config)
        )
    logging.info(output)


if __name__ == "__main__":
    sample_run()
        