from apache_beam.io import filesystems
from datetime import datetime
from scipy.signal import butter, lfilter, find_peaks, sosfilt

import apache_beam as beam
import io
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from config import load_pipeline_config


config = load_pipeline_config()


class BaseSift(beam.PTransform): 
    """
    All our Sift transforms will have the structure:
    - input: full audio signal
    - output: timeframes
    """
    name = "BaseSift"

    # general params
    debug       = config.general.debug
    sample_rate = config.audio.source_sample_rate

    # sift-specific params
    max_duration    = config.sift.max_duration
    window_size     = config.sift.window_size
    
    # plot params
    plot                = config.sift.plot
    plot_path_template  = config.sift.plot_path_template

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
        signal, start, end, encounter_ids = pcoll
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
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        logging.info(f"Postprocessing {self.name} sifted signal.")
        logging.debug(f"Signal: {signal}")
        logging.debug(f"Min-max detections: {min_max_detections}")
        logging.debug(f"Key: {key}")

        global_detection_range = [
            min_max_detections[key]["min"], 
            min_max_detections[key]["max"]
        ]

        return signal[global_detection_range[0]:global_detection_range[-1]], start, end, encounter_ids

    def _plot_signal_detections(self, pcoll, min_max_detections, all_detections, params=None):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)
        
        min_max_detection_samples = [
            min_max_detections[key]["min"],  # maually fix ordering
            min_max_detections[key]["max"]
        ]
        logging.info(f"Plotting signal detections: {min_max_detection_samples}")
        
        # datetime format matches original audio file name
        plot_path = self.plot_path_template.format(
            sift=self.name,
            year=start.year,
            month=start.month,
            day=start.day,
            plot_name=key
        )

        # TODO make dirs cleaner
        if not os.path.isdir(os.path.sep.join(plot_path.split(os.path.sep)[:-1])):
            os.makedirs(os.path.sep.join(plot_path.split(os.path.sep)[:-1]))

        # normalize and center
        signal = signal / np.max(signal)    # normalize
        signal = signal - np.mean(signal)   # center

        plt.figure(figsize=(20, 10))
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
        title += f"Params: {params} \n" if params else "" 
        title += f"Encounters: {encounter_ids}"
        plt.title(title) 
        plt.savefig(plot_path)
        plt.show()


class Butterworth(BaseSift):
    name = "Butterworth"

    def __init__(
            self,
            lowcut: int = None,
            highcut: int = None,
            order: int = None,
            output: str = None,
            sift_threshold: float = None,
        ):
        super().__init__()

        # define bandpass
        self.lowcut  = config.sift.butterworth.lowcut if not lowcut else lowcut
        self.highcut = config.sift.butterworth.highcut if not highcut else highcut
        self.order   = config.sift.butterworth.order if not order else order
        self.output  = config.sift.butterworth.output if not output else output

        # apply bandpass
        self.sift_threshold  = (
            config.sift.butterworth.sift_threshold 
            if not sift_threshold else sift_threshold
        )

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
        sifted_output       = pcoll         | "Postprocess"         >> beam.Map(
            self._postprocess, 
            min_max_detections=beam.pvalue.AsDict(min_max_detections), 
        )

        # plots for debugging purposes
        if self.debug or self.plot:
            pcoll | "Plot Sifted Output" >> beam.Map(
                self._plot_signal_detections, 
                min_max_detections=beam.pvalue.AsDict(min_max_detections), 
                all_detections=beam.pvalue.AsDict(all_detections),
                params={
                    "lowcut": self.lowcut,
                    "highcut": self.highcut,
                    "order": self.order,
                    "threshold": self.sift_threshold,
                }
            )

        return sifted_output

    def _butter_bandpass(self):
        """
        Returns specific Butterworth filter (IIR) from parameters in config.sift.butterworth
        """
        nyq = 0.5 * self.sample_rate
        low = self.lowcut / nyq
        high = self.highcut / nyq
        return butter(self.order, [low, high], btype='band', output=self.output)
    
    def _frequency_filter_sift(
            self,
            batch: tuple,
        ):
        key, signal = batch

        logging.info(f"Start frequency detection on (key, signal): {(key, signal.shape)}")

        # Apply bandpass filter
        butter_coeffients = self._butter_bandpass()
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
        peaks, _ = find_peaks(energy, height=self.sift_threshold)
        logging.debug(f"Peaks: {peaks}")

        # Convert peak indices to time
        peak_samples = peaks * (self.window_size)
        logging.debug(f"Peak samples: {peak_samples}")

        yield (key, peak_samples)


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
    
