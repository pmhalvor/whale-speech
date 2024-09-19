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

    sample_rate     = config.audio.source_sample_rate
    max_duration    = config.sift.max_duration
    window_size     = config.sift.window_size
    plot            = config.sift.plot

    def __init__(self, args: dict):
        self.start = datetime.strptime(args.get('start'), "%Y-%m-%dT%H:%M:%S")
        self.end = datetime.strptime(args.get('end'), "%Y-%m-%dT%H:%M:%S")

    def _preprocess(self, pcoll):
        """
        pcoll: tuple(audio, start_time, end_time, row.encounter_ids)
        """
        signal = pcoll[0]

        max_samples = self.max_duration * self.sample_rate
        if signal.shape[0] > max_samples:
            signal_batches = np.array_split(signal, signal.shape[0] // max_samples)
            for batch in signal_batches:
                yield batch
        else:
            yield signal

    def _postprocess(self, pcoll, min_max_detections):
        signal = pcoll[0]

        logging.info(f"Postprocessing {self.name} sifted signal.")
        logging.info(f"Signal: {signal}")
        logging.info(f"Min-max detections: {min_max_detections}")
        first_detection = min_max_detections["peak"]["min"]
        last_detection = min_max_detections["peak"]["max"]
        global_detection_range = self._seconds_to_sample([first_detection, last_detection])


        return signal[global_detection_range[0]:global_detection_range[-1]]

    def _global_peaks(self, peak_times):
        return [min(peak_times), max(peak_times)]

    def _seconds_to_sample(self, seconds):
        return np.array([int(s*self.sample_rate) for s in seconds])
    
    def _plot_signal_detections(self, element, min_max_detections, all_detections):
        signal, start, end, encounter_ids = element
        # start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        # end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
        
        logging.info(f"min-max detections: {min_max_detections}")
        min_max_detection_times = [
            min_max_detections["peak"]["min"],  # maually fix ordering
            min_max_detections["peak"]["max"]
        ]
        min_max_detection_samples = self._seconds_to_sample(min_max_detection_times)

        logging.info(f"All detections: {all_detections}")
        all_detection_samples = self._seconds_to_sample(all_detections["peak"])

        logging.info(f"Plotting signal detections: {min_max_detection_samples}")
        
        # datetime format matches original audio file name
        plot_name = f"{start.strftime('%Y%m%dT%H%M%S')}-{end.strftime('%H%M%S')}"
        plot_path = config.sift.plot_path_template.format(
            sift_type=self.name,
            year=start.year,
            month=start.month,
            day=start.day,
            plot_name=plot_name
        )

        # TODO make dirs cleaner
        if not os.path.isdir(os.path.sep.join(plot_path.split(os.path.sep)[:-1])):
            os.makedirs(os.path.sep.join(plot_path.split(os.path.sep)[:-1]))

        # normalize signal
        signal = signal / np.max(signal)

        plt.figure(figsize=(20, 10))
        plt.plot(signal)
        if len(min_max_detection_samples):
            # shade window that resulted in detection
            plt.axvspan(
                min_max_detection_samples[0],
                min_max_detection_samples[-1],
                alpha=0.2,
                color='y',
                zorder=5, # bring to front
            )

        if len(all_detection_samples):
            # shade window that resulted in detection
            for detection in all_detection_samples:
                plt.axvspan(
                    detection - self.window_size/2,
                    detection + self.window_size/2,
                    alpha=0.7,
                    color='r',
                    zorder=7, # bring to front
                )

        plt.legend(['Input signal', 'detection window', 'all detections'])
        plt.title(f"Signal detections: {plot_name} ({self.name}) Encounters: {encounter_ids}")
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
    peak_threshold  = config.sift.butterworth.peak_threshold

    def expand(self, pcoll):
        """
        pcoll: tuple(audio, start_time, end_time, row.encounter_ids)
        """

        batch           = pcoll         | "Preprocess"      >> beam.ParDo(self._preprocess)

        # batched process
        detections      = batch         | "Sift Frequency"  >> beam.ParDo(self._frequency_filter_sift)
        logging.info(f"Found {detections} detections in {self.name}.")
        
        # BUG currently mixing peak values for all geo search results. Only want to look at detections for this specific audio file
        min_max_detections  = detections | "Min-Max Detection"  >> beam.CombinePerKey(MinMax())
        all_detections      = detections | "List Detections"    >> beam.CombinePerKey(ListCombine())
        
        # full-input process
        sifted_output   = pcoll         | "Postprocess" >> beam.Map(
            self._postprocess, 
            min_max_detections=beam.pvalue.AsDict(min_max_detections), 
        )

        # for debugging purposes
        if config.general.debug or config.sift.plot:
            pcoll | "Plot Sifted Output" >> beam.Map(
                self._plot_signal_detections, 
                min_max_detections=beam.pvalue.AsDict(min_max_detections), 
                all_detections=beam.pvalue.AsDict(all_detections),
            )
            # detections       | "Write Peak Times"    >> beam.io.WriteToText("peak_times.txt")
            # global_peaks        | "Write Global Peaks"  >> beam.io.WriteToText("global_peaks.txt")

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
            signal,
        ):
        logging.info(f"Start frequency detection on signal: {signal}")

        # Apply bandpass filter
        butter_coeffients = self._butter_bandpass()
        if self.output == "ba":
            filtered_signal = lfilter(butter_coeffients[0], butter_coeffients[1], signal)

        if self.output == "sos":
            filtered_signal = sosfilt(butter_coeffients, signal)

        logging.info(f"Filtered signal: {filtered_signal}")

        # Calculate energy in windows
        energy = np.array([
            sum(abs(filtered_signal[i:i+self.window_size]**2))
            for i in range(0, len(filtered_signal), self.window_size)
        ])

        logging.info(f"Energy: {energy}")

        # Normalize energy
        energy = energy / np.max(energy)

        logging.info(f"Normalized energy: {energy}")

        # Find peaks above threshold
        peaks, _ = find_peaks(energy, height=self.peak_threshold)

        logging.info(f"Peaks: {peaks}")

        # Convert peak indices to time
        peak_times = peaks * (self.window_size / self.sample_rate)

        logging.info(f"Peak times: {peak_times}")

        yield ("peak", peak_times)



class ListCombine(beam.CombineFn):
    def create_accumulator(self):
        return []

    def add_input(self, accumulator, input):
        logging.info(f"Adding input {input} to accumulator. type: {type(input)}")
        if isinstance(input, np.ndarray):
            input = input.tolist()
        accumulator += input
        return accumulator

    def merge_accumulators(self, accumulators):
        return [item for sublist in accumulators for item in sublist]

    def extract_output(self, accumulator):
        return accumulator
    

class MinMax(ListCombine):
    def extract_output(self, accumulator):
        if not accumulator:
            return None

        logging.info(f"Extracting min-max from accumulator {accumulator}")
        logging.info(f"type: {type(accumulator)}")
        return {"min": min(accumulator), "max": max(accumulator)}
    
