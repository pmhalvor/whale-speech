from apache_beam.io import filesystems
from datetime import datetime
from scipy.signal import butter, lfilter, find_peaks, sosfilt

import apache_beam as beam
import io
import logging
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

    def __init__(self, args: dict):
        self.start = datetime.strptime(args.get('start'), "%Y-%m-%dT%H:%M:%S")
        self.end = datetime.strptime(args.get('end'), "%Y-%m-%dT%H:%M:%S")

    def _build_key(
            self,
            start_time: datetime,
            end_time: datetime,
            encounter_ids: list,
        ):
        start_str = start_time.strftime('%Y%m%dT%H%M%S')
        end_str = end_time.strftime('%H%M%S')
        encounter_str = "_".join(encounter_ids)
        return f"{start_str}_{end_str}_{encounter_str}"

    def _preprocess(self, pcoll):
        """
        pcoll: tuple(audio, start_time, end_time, row.encounter_ids)
        """
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        max_samples = self.max_duration * self.sample_rate
        if signal.shape[0] > max_samples:
            signal_batches = np.array_split(signal, signal.shape[0] // max_samples)
            for batch in signal_batches:
                yield batch
        else:
            yield (key, signal)

    def _postprocess(self, pcoll, min_max_detections):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)

        logging.info(f"Postprocessing {self.name} sifted signal.")
        logging.info(f"Signal: {signal}")
        logging.info(f"Min-max detections: {min_max_detections}")
        logging.info(f"Key: {key}")

        first_detection = min_max_detections[key]["min"]
        last_detection = min_max_detections[key]["max"]
        global_detection_range = self._seconds_to_sample([first_detection, last_detection])

        return signal[global_detection_range[0]:global_detection_range[-1]]

    def _seconds_to_sample(self, seconds):
        return np.array([int(s*self.sample_rate) for s in seconds])
    
    def _plot_signal_detections(self, pcoll, min_max_detections, all_detections, params=None):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)
        
        logging.info(f"min-max detections: {min_max_detections}")
        min_max_detection_times = [
            min_max_detections[key]["min"],  # maually fix ordering
            min_max_detections[key]["max"]
        ]
        min_max_detection_samples = self._seconds_to_sample(min_max_detection_times)

        logging.info(f"All detections: {all_detections}")
        all_detection_samples = self._seconds_to_sample(all_detections[key])

        logging.info(f"Plotting signal detections: {min_max_detection_samples}")
        
        # datetime format matches original audio file name
        plot_path = self.plot_path_template.format(
            sift_type=self.name,
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
        
        
        if len(all_detection_samples):
            # shade window that resulted in detection
            for detection in all_detection_samples:
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

    # define bandpass
    lowcut  = config.sift.butterworth.lowcut
    highcut = config.sift.butterworth.highcut
    order   = config.sift.butterworth.order
    output  = config.sift.butterworth.output
    
    # apply bandpass
    threshold  = config.sift.butterworth.threshold

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
                    "threshold": self.threshold,
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

        logging.info(f"Start frequency detection on (key, signal): {(key, signal)}")

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
        peaks, _ = find_peaks(energy, height=self.threshold)
        logging.debug(f"Peaks: {peaks}")

        # Convert peak indices to time
        peak_times = peaks * (self.window_size / self.sample_rate)
        logging.debug(f"Peak times: {peak_times}")

        yield (key, peak_times)


class ListCombine(beam.CombineFn):
    name = "ListCombine"

    def create_accumulator(self):
        return []

    def add_input(self, accumulator, input):
        """
        Key is not available in this method, 
        though inputs are only added to accumulator under correct key.
        """
        logging.info(f"Adding input {input} to {self.name} accumulator.")
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
    
