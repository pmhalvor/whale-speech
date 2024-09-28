import apache_beam as beam

from datetime import datetime
from typing import Dict, Any
from types import SimpleNamespace

import librosa
import logging
import numpy as np

import requests
import math 


logging.getLogger().setLevel(logging.INFO)


class BaseClassifier(beam.PTransform): 
    name = "BaseClassifier"


    def __init__(self, config: Dict[str, Any]):
        self.source_sample_rate = config.audio.source_sample_rate

        self.batch_duration     = config.classify.batch_duration
        self.model_sample_rate  = config.classify.model_sample_rate
        self.model_url          = config.classify.model_url

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


class WhaleClassifier(BaseClassifier):
    def expand(self, pcoll):
        key_batch       = pcoll             | "Preprocess"              >> beam.ParDo(self._preprocess)
        batched_outputs = key_batch         | "Classify"                >> beam.ParDo(InferenceClient(self.model_url))
        grouped_outputs = batched_outputs   | "Combine batched_outputs" >> beam.CombinePerKey(ListCombine())
        outputs         = pcoll             | "Postprocess"             >> beam.Map(
            self._postprocess,
            grouped_outputs=beam.pvalue.AsDict(grouped_outputs), 
        )
        return outputs
    
    def _postprocess(self, pcoll, grouped_outputs):
        signal, start, end, encounter_ids = pcoll
        key = self._build_key(start, end, encounter_ids)
        output = grouped_outputs.get(key, [])

        logging.info(f"Postprocessing {key} with signal {len(signal)} and output {len(output)}")

        return signal, start, end, encounter_ids, output
    

class InferenceClient(beam.DoFn):
    def __init__(self, model_url: str):
        self.model_url = model_url

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
        
        response = requests.post(self.model_url, json=data)
        response.raise_for_status()

        yield response.json()


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
        audio = SimpleNamespace(source_sample_rate=16_000),
        classify = SimpleNamespace(
            batch_duration=30, # seconds
            model_sample_rate=10_000,
            model_url="http://127.0.0.1:5000/predict"
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
        