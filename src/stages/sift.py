from apache_beam.io import filesystems
from datetime import datetime
from scipy.signal import butter, lfilter, find_peaks, sosfilt
from types import SimpleNamespace
from typing import Dict, Any

import apache_beam as beam
import io
import logging
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from config import load_pipeline_config


config = load_pipeline_config()


class BaseSift(beam.PTransform): 
    """
    All our Sift transforms will have the structure:
    - input: full audio signal
    - output: timeframes
    """
    name = "BaseSift"

    def __init__(self, config: SimpleNamespace):
        # general params
        self.debug = config.general.debug
        self.is_local = config.general.is_local
        self.sample_rate = config.audio.source_sample_rate
        self.store = config.sift.store_sift_audio

        # sift-specific params
        self.max_duration = config.sift.max_duration
        self.threshold = None
        self.window_size = config.sift.window_size
        
        # plotting params
        self.plot = config.sift.plot
        self.plot_path_template = config.sift.plot_path_template
        self.show_plots = config.general.show_plots

        # store params
        self.output_array_path_template = config.sift.output_array_path_template
        self.output_table_path_template = config.sift.output_table_path_template
        self.params_path_template = None  # specific to each sift
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.sift.sift_table_id
        self.temp_location = config.general.temp_location
        self.schema = self._schema_to_dict(config.sift.sift_table_schema)
        self.workbucket = config.general.workbucket
        self.write_params = config.bigquery.__dict__


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

    def _get_filter_params(self):
        return {}

    def _get_path_params(self):
        return {"name": self.name.lower()}

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

    def _preprocess(self, pcoll):
        """
        pcoll: tuple(audio, start_time, end_time, row.encounter_ids)
        """
        signal, start, end, encounter_ids, _ = pcoll
        key = self._build_key(start, end, encounter_ids)

        max_samples = self.max_duration * self.sample_rate

        if signal.shape[0] > max_samples:
            logging.debug(f"Signal size exceeds max sample size {max_samples}.")
            split_indices = [max_samples*(i+1) for i  in range(math.floor(signal.shape[0] / max_samples))]
            signal_batches = np.array_split(signal, split_indices)
            logging.debug(f"Split signal into {len(signal_batches)} batches of size {max_samples}.")
            logging.debug(f"Size fo final batch {len(signal_batches[1])}")

            for batch in signal_batches:
                yield (key, batch)
        else:
            yield (key, signal)

    def _postprocess(self, pcoll, min_max_detections):
        signal, start, end, encounter_ids, _ = pcoll
        key = self._build_key(start, end, encounter_ids)

        logging.info(f"Postprocessing {self.name} sifted signal.")
        logging.debug(f"Signal: {signal}")
        logging.debug(f"Min-max detections: {min_max_detections}")
        logging.debug(f"Key: {key}")

        global_detection_range = [
            min_max_detections[key]["min"], 
            min_max_detections[key]["max"]
        ]

        sifted_signal = signal[global_detection_range[0]:global_detection_range[-1]]
        audio_path = "No sift audio path stored."
        detections_path = "No detections path stored."

        return [(sifted_signal, start, end, encounter_ids, audio_path, detections_path)]

    def _plot_signal_detections(self, pcoll, min_max_detections, all_detections):
        signal, start, end, encounter_ids, _ = pcoll
        key = self._build_key(start, end, encounter_ids)
        
        min_max_detection_samples = [
            min_max_detections[key]["min"],  # maually fix ordering
            min_max_detections[key]["max"]
        ]
        logging.info(f"Plotting signal detections: {min_max_detection_samples}")
        
        # datetime format matches original audio file name
        params_path = self.params_path_template.format(
            **self._get_path_params()
        )
        plot_path = self.plot_path_template.format(
            params=params_path,
            key=key
        )

        if not beam.io.filesystems.FileSystems.exists(os.path.dirname(plot_path)):
            beam.io.filesystems.FileSystems.mkdirs(os.path.dirname(plot_path))

        # normalize and center
        signal = signal / np.max(signal)    # normalize
        signal = signal - np.mean(signal)   # center

        # plt.figure(figsize=(20, 10))
        fig = plt.figure()
        plt.plot(signal)
        
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
        
        if len(all_detections[key]):
            # shade window that resulted in detection
            for detection in all_detections[key]:
                plt.axvspan(
                    detection - self.window_size/2,
                    detection + self.window_size/2,
                    alpha=0.5,
                    color='r',
                    zorder=5, # on top of signal
                )


        plt.legend(['Input signal', 'detection window', 'all detections']).set_zorder(10)
        plt.xlabel(f'Samples (seconds * {self.sample_rate} Hz)')
        plt.ylabel('Amplitude (normalized and centered)') 

        title = f"({self.name}) Signal detections: {start.strftime('%Y-%m-%d %H:%M:%S')}-{end.strftime('%H:%M:%S')}\n"
        title += f"Params: {self._get_path_params()} \n" if self._get_path_params() else "" 
        title += f"Encounters: {encounter_ids}"
        plt.title(title) 
        plt.savefig(plot_path)

        plt.show() if self.show_plots else plt.close()


class Butterworth(BaseSift):
    name = "Butterworth"

    def __init__(
            self,
            config: SimpleNamespace
        ):
        super().__init__(config)

        # define bandpass
        self.lowcut  = config.sift.butterworth.lowcut
        self.highcut = config.sift.butterworth.highcut
        self.order   = config.sift.butterworth.order
        self.output  = config.sift.butterworth.output

        # apply bandpass
        self.threshold  = config.sift.butterworth.sift_threshold 

        # store params
        self.params_path_template = config.sift.butterworth.params_path_template

    @staticmethod
    def _butter_bandpass(
        sample_rate: int,
        lowcut: float,
        highcut: float,
        order: int,
        output
    ):
        """
        Returns specific Butterworth filter (IIR) from parameters in config.sift.butterworth
        """
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band', output=output)
    
    def expand(self, pcoll):
        """
        pcoll: tuple(audio, start_time, end_time, row.encounter_ids)
        """
        # full-input preprocess
        batch               = pcoll         | "Preprocess"          >> beam.ParDo(self._preprocess)

        # batched process
        detections          = batch         | "Sift Frequency"      >> beam.ParDo(self._frequency_filter_sift)
        min_max_detections  = detections    | "Min-Max Detections"  >> beam.CombinePerKey(MinMaxCombine())
        all_detections      = detections    | "List Detections"     >> beam.CombinePerKey(ListCombine())
        
        # full-input postprocess
        sifted_output       = pcoll         | "Postprocess"         >> beam.ParDo(
            self._postprocess, 
            min_max_detections=beam.pvalue.AsDict(min_max_detections), 
        )

        if self.store:
            results = sifted_output | "Store Sifted Audio" >> beam.ParDo(
                self._store,
                detections=beam.pvalue.AsDict(all_detections),
            )
        else:
            results = sifted_output

        # plots for debugging purposes
        if self.debug or self.plot:
            pcoll | "Plot Sifted Output" >> beam.Map(
                self._plot_signal_detections, 
                min_max_detections=beam.pvalue.AsDict(min_max_detections), 
                all_detections=beam.pvalue.AsDict(all_detections),
            )

        return results

    def _frequency_filter_sift(
            self,
            batch: tuple,
        ):
        key, signal = batch
        filter_params = self._get_filter_params()

        logging.info(f"Start frequency detection on (key, signal): {(key, signal.shape)}")

        # Apply bandpass filter
        butter_coeffients = self._butter_bandpass(
            self.sample_rate,
            **filter_params,
        )
        if self.output == "ba":
            filtered_signal = lfilter(butter_coeffients[0], butter_coeffients[1], signal)
        elif self.output == "sos":
            filtered_signal = sosfilt(butter_coeffients, signal)
        else:
            raise ValueError(f"Invalid output type for config.sift.butterworth.output: {self.output}")
        logging.debug(f"Filtered signal: {filtered_signal}")

        # Calculate energy in windows
        energy = np.array([
            sum(abs(filtered_signal[i:i+self.window_size]**2))
            for i in range(0, len(filtered_signal), self.window_size)
        ])
        logging.debug(f"Energy: {energy}")

        # Normalize energy
        energy = energy / np.max(energy)
        logging.debug(f"Normalized energy: {energy}")

        # Find peaks above threshold
        peaks, _ = find_peaks(energy, height=self.threshold)
        logging.debug(f"Peaks: {peaks}")

        # Convert peak indices to time
        peak_samples = peaks * (self.window_size)
        logging.debug(f"Peak samples: {peak_samples}")

        yield (key, peak_samples)

    def _get_filter_params(self):
        return {
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "output": self.output,
        }

    def _get_path_params(self):
        path_params = self._get_filter_params()
        path_params.pop("output")
        path_params["threshold"] = self.threshold
        path_params["name"] = self.name.lower()
        return path_params

    def _get_export_paths(self, key):
        params_path = self.params_path_template.format(**self._get_path_params())
        audio_path = self.output_array_path_template.format(
            params=params_path,
            key=key,
            filename="audio.npy"
        )
        detections_path = self.output_array_path_template.format(
            params=params_path,
            key=key,
            filename="detections.npy"
        )

        if not self.is_local:
            audio_path = os.path.join(self.workbucket,audio_path)
            detections_path = os.path.join(self.workbucket,detections_path)

        return audio_path, detections_path

    def _store(self, sifted_output, detections):
        signal, start, end, encounter_ids, audio_path, detections_path  = sifted_output
        logging.info(f"Signal shape: {signal.shape}")
        logging.info(f"Start: {start}")
        logging.info(f"End: {end}")
        logging.info(f"Encounter IDs: {encounter_ids}")

        key = self._build_key(start, end, encounter_ids)

        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        
        if signal.shape[0] == 0:
            logging.info(f"Empty sifted signal for {key}.")
            return [(signal, start, end, encounter_ids, audio_path, detections_path)]

        audio_path, detections_path = self._get_export_paths(key)

        # upload metadata to table
        if self.is_local:
            self._store_local(
                key, 
                audio_path, 
                detections_path,
                self._get_path_params(),
            )
        else:
            self._store_bigquery(
                key, 
                audio_path, 
                detections_path,
                self._get_path_params(),
            )

        # store sifted audio
        with filesystems.FileSystems.create(audio_path) as f:
            logging.info(f"Storing sifted audio to {audio_path}")
            np.save(f, signal)

        # store detections
        with filesystems.FileSystems.create(detections_path) as f:
            logging.info(f"Storing detections to {detections_path}")    
            np.save(f, detections[key])

        logging.info(f"Stored sifted audio and detections for {key}.")
        return [(signal, start, end, encounter_ids, audio_path, detections_path)]

    def _store_local(
        self, 
        key: str,
        audio_path: str, 
        detections_path: str, 
        params: Dict[str, Any], 
    ):
        table_path = self.output_table_path_template.format(table_id=self.table_id)

        if not filesystems.FileSystems.exists(table_path):
            # build parent dir if necessary 
            filesystems.FileSystems.create(table_path)
            df = pd.DataFrame([{
                "key": key,
                "sift_audio_path": audio_path,
                "sift_detections_path": detections_path,
                "params": json.dumps(params)
            }])
            df.to_json(table_path, index=False, orient="records")
        else:
            df = pd.read_json(table_path, orient="records")
            new_row = pd.DataFrame([{
                "key": key,
                "sift_audio_path": audio_path,
                "sift_detections_path": detections_path,
                "params": json.dumps(params)
            }])
            df = pd.concat([df, new_row])
            df = df.drop_duplicates()
            df.to_json(table_path, index=False, orient="records")

    def _store_bigquery(
        self, 
        key: str,
        audio_path: str, 
        detections_path: str, 
        params: Dict[str, Any], 
    ):
        [{
            "key": key,
            "sift_audio_path": audio_path,
            "sift_detections_path": detections_path,
            "params": json.dumps(params)
        }] | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            self.table_id,
            dataset=self.dataset_id,
            project=self.project,
            schema=self.schema,
            custom_gcs_temp_location=self.temp_location,
            **self.write_params
        )


class ListCombine(beam.CombineFn):
    name = "ListCombine"

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
    

class MinMaxCombine(ListCombine):
    name = "MinMaxCombine"

    def extract_output(self, accumulator):
        if not accumulator:
            return None

        logging.debug(f"Extracting min-max output from {accumulator}")
        return {"min": min(accumulator), "max": max(accumulator)}
    
