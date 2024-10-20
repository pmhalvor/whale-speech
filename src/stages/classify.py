from apache_beam.io import filesystems
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
import pandas as pd
import pickle

import requests
import math 
import matplotlib.pyplot as plt
import scipy.signal


logging.getLogger().setLevel(logging.INFO)


class BaseClassifier(beam.PTransform): 
    name = "BaseClassifier"

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.filesystem = config.general.filesystem.lower()
        self.source_sample_rate = config.audio.source_sample_rate

        self.batch_duration     = config.classify.batch_duration
        self.model_sample_rate  = config.classify.model_sample_rate
        self.inference_url      = config.classify.inference_url

        # plotting parameters
        self.hydrophone_sensitivity = config.classify.hydrophone_sensitivity
        self.med_filter_size        = config.classify.med_filter_size
        self.plot_scores            = config.classify.plot_scores
        self.plot_path_template     = config.classify.plot_path_template
        self.show_plots             = config.general.show_plots

        # store parameters
        self.store = config.classify.store_classifications
        self.output_array_path_template = config.classify.output_array_path_template
        self.output_table_path_template = config.classify.output_table_path_template
        
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.classify.classification_table_id
        self.schema = self._schema_to_dict(config.classify.classification_table_schema)
        self.temp_location = config.general.temp_location
        self.workbucket = config.general.workbucket
        self.write_params = config.bigquery.__dict__
        
        # TODO dynamically update params used in filter to build classification path
        self.params_path_template = config.sift.butterworth.params_path_template
        self.path_params = {
            "name": "butterworth",
            "lowcut": config.sift.butterworth.lowcut,
            "highcut": config.sift.butterworth.highcut,
            "order": config.sift.butterworth.order,
            "threshold": config.sift.butterworth.sift_threshold,
        } 

    @staticmethod
    def _schema_to_dict(schema):
        return {
            "fields": [
                {
                    "name": name, 
                    "type": getattr(schema, name).type, 
                    "mode": getattr(schema, name).mode
                } 
                for name in vars(schema)
            ]
        }

    def _preprocess(self, pcoll):
        signal, start, end, encounter_ids, _, _ = pcoll
        key = self._build_key(start, end, encounter_ids)
        logging.info(f"Classifying {key} with signal shape {signal.shape}")

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
        plt.title('Model Scores')

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
        logging.debug(f"time: {t.shape}, freq: {f.shape}, psd: {psd.shape}")

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

    def _plot_audio(self, audio, start, key):
        try:
            with open(f"data/plots/Butterworth/{start.year}/{start.month}/{start.day}/data/{key}_min_max.pkl", "rb") as f:
                min_max_samples = pickle.load(f)
            with open(f"data/plots/Butterworth/{start.year}/{start.month}/{start.day}/data/{key}_all.pkl", "rb") as f:
                all_samples = pickle.load(f)
        except FileNotFoundError:
            min_max_samples = []
            all_samples = []

        def _plot_signal_detections(min_max_detection_samples, all_samples):
            # TODO refactor plot_signal_detections in classify 
            logging.info(f"Plotting signal detections: {min_max_detection_samples}")
            
            # NOTE: for legend logic, plot min_max window first
            if len(min_max_detection_samples):
                # shade window that resulted in detection
                plt.axvspan(
                    min_max_detection_samples[0],
                    min_max_detection_samples[-1],
                    alpha=0.3,
                    color='y',
                    zorder=7, # on top of all detections
                )
            
            if len(all_samples):
                # shade window that resulted in detection
                for detection in all_samples:
                    plt.axvspan(
                        detection - self.config.sift.window_size/2,
                        detection + self.config.sift.window_size/2,
                        alpha=0.5,
                        color='r',
                        zorder=5, # on top of signal
                    )

            plt.legend(['Input signal', 'detection window', 'all detections']).set_zorder(10)
            plt.xlabel(f'Samples (seconds * {self.config.audio.source_sample_rate} Hz)') 
            plt.ylabel('Amplitude (normalized and centered)') 

            title = f"Signal detections: {start.strftime('%Y-%m-%d %H:%M:%S')}"
            plt.title(title) 

        plt.plot(audio)
        plt.xlabel(f'Samples (seconds * {self.config.audio.source_sample_rate} Hz)') 
        plt.ylabel('Amplitude (normalized and centered)') 
        _plot_signal_detections(min_max_samples, all_samples)

    def _plot(self, output):
        audio, start, end, encounter_ids, scores, _ = output
        key = self._build_key(start, end, encounter_ids)

        if len(audio) == 0:
            logging.info("No audio to classify and plot")
            logging.debug("(i.e. no detections from sift-stage)")
            return
        else:
            logging.info(f"Plotting {key} with audio shape {audio.shape} and scores shape {len(scores)}")
        
        fig = plt.figure(figsize=(24, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

        # Plot spectrogram:
        plt.subplot(gs[0])
        self._plot_audio(audio, start, key)

        # Plot spectrogram:
        plt.subplot(gs[1])
        t, _, _ = self._plot_spectrogram_scipy(audio)

        # Plot scores:
        fig.add_subplot(gs[2])
        self._plot_scores(scores, t=t)

        plt.tight_layout()

        plot_path = self.plot_path_template.format(
            params=self._get_params_path(),
            plot_name=key
        )
        filesystems.FileSystems.create(plot_path)

        plt.savefig(plot_path)
        plt.show() if self.show_plots else plt.close()

    def _get_params_path(self):
        return self.params_path_template.format(
            **self.path_params
        )

    def _get_export_path(self, key):
        export_path = self.output_array_path_template.format(
            params=self._get_params_path(),
            key=key
        )

        if self.filesystem == "gcp":
            export_path = os.path.join(self.workbucket, export_path)

        return export_path

    def _store(self, outputs):
        audio, start, end, encounter_ids, scores, no_path = outputs
        key = self._build_key(start, end, encounter_ids)

        if len(scores) == 0 or len(audio) == 0:
            logging.info(f"No detections for {key}. Skipping storage.")
            return [(audio, start, end, encounter_ids, scores, no_path)]

        classifications_path = self._get_export_path(key)

        # update metadata table
        if self.filesystem == "local":
            self._store_local(key, classifications_path)
        elif self.filesystem == "gcp":
            self._store_bigquery(key, classifications_path)
        else:
            raise ValueError(f"Unsupported filesystem: {self.filesystem}")

        # store classifications
        with filesystems.FileSystems.create(classifications_path) as f:
            logging.info(f"Storing sifted audio to {classifications_path}")
            np.save(f, np.array(scores).flatten())

        return [(audio, start, end, encounter_ids, scores, classifications_path)]

    def _store_local(self, key, classifications_path):
        logging.info(f"Storing local classification for {key}")
        
        table_path = self.output_table_path_template.format(table_id=self.table_id)

        if not filesystems.FileSystems.exists(table_path):
            filesystems.FileSystems.create(table_path)
            df = pd.DataFrame([{
                "key": key,
                "classifications_path": classifications_path,
            }])
            df.to_json(table_path, index=False, orient="records")
        else:
            df = pd.read_json(table_path, orient="records")
            new_row = pd.DataFrame([{
                "key": key,
                "classifications_path": classifications_path,
            }])
            df = pd.concat([df, new_row])
            df = df.drop_duplicates()
            df.to_json(table_path, index=False, orient="records")

    def _store_bigquery(
            self, 
            key, 
            classifications_path
        ):
        logging.info(f"Storing classification for {key} in BigQuery")
        [{
            "key": key,
            "classifications_path": classifications_path
        }] | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            self.table_id,
            dataset=self.dataset_id,
            project=self.project,
            schema=self.schema,
            custom_gcs_temp_location=self.temp_location,
            **self.write_params
        )


class WhaleClassifier(BaseClassifier):
    def expand(self, pcoll):
        key_batch       = pcoll             | "Preprocess"              >> beam.ParDo(self._preprocess)
        batched_outputs = key_batch         | "Classify"                >> beam.ParDo(InferenceClient(self.config))
        grouped_outputs = batched_outputs   | "Combine batched_outputs" >> beam.CombinePerKey(ListCombine())
        outputs         = pcoll             | "Postprocess"             >> beam.ParDo(
            self._postprocess,
            grouped_outputs=beam.pvalue.AsDict(grouped_outputs), 
        )

        if self.store:
            outputs = outputs | "Store classifications" >> beam.ParDo(self._store)

        if self.plot_scores:
            outputs | "Plot scores" >> beam.Map(self._plot)

        logging.info(f"Finished {self.name} stage: {outputs}")
        return outputs
    
    def _postprocess(self, pcoll, grouped_outputs):
        signal, start, end, encounter_ids, _, _ = pcoll
        key = self._build_key(start, end, encounter_ids)
        scores = grouped_outputs.get(key, [])
        logging.info(f"Postprocessing {key} with signal {len(signal)} and scores {len(scores)}")

        return [(signal, start, end, encounter_ids, scores, "No classification path stored.")]


class InferenceClient(beam.DoFn):
    def __init__(self, config: SimpleNamespace):

        self.inference_url = config.classify.inference_url
        self.retries = config.classify.inference_retries

    def process(self, element):
        key, batch = element
        logging.info(f"Sending batch {key} with {len(batch)} samples to inference at: {self.inference_url}")

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
                response = requests.post(self.inference_url, json=data)
                response.raise_for_status()
                break
            except requests.exceptions.ConnectionError as e:
                logging.info(f"Connection error: {e}")
                logging.info(f"Retrying in {wait*wait} seconds.")
                wait += 1
                time.sleep(wait*wait)

        response = requests.post(self.inference_url, json=data)
        response.raise_for_status()

        predictions = response.json().get("predictions", [])

        logging.info(f"Inference response:\n key: {key}  predictions:{len(predictions)}")

        yield (key, predictions)  # TODO fix mixing yield and return in DoFn


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
            inference_url="http://127.0.0.1:5000/predict",
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
        