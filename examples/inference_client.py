import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, DirectOptions, GoogleCloudOptions, StandardOptions
from apache_beam.utils.shared import Shared

from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.transforms.embeddings.tensorflow_hub import _TensorflowHubModelHandler
from apache_beam.ml.inference.base import RunInference

from typing import Dict, Any, Optional, Callable

import logging
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

import requests

import psutil


logging.getLogger().setLevel(logging.INFO)


def print_available_ram():
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert from bytes to GB
    total_ram = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Total RAM: {total_ram:.2f} GB")

print_available_ram()


class InferenceClient(beam.DoFn):
    def __init__(self, client_url: str,):
        self.client_url = client_url

    def process(self, element):
        key, batch = element

        data = {
            "key": key,
            "batch": batch.numpy().tolist(),
        }
        
        response = requests.post(self.client_url, json=data)
        response.raise_for_status()

        yield response.json()

class PostProcess(beam.DoFn):
    def process(self, element):
        key, result = element

        logging.info(f"key: {key}")
        logging.info(f"result: {result}")

        yield key, result


def simple_run():
    signal = np.load("data/audio/butterworth/2016/12/20161221T004930-005030-9182.npy")
    waveform_exp = tf.cast(tf.expand_dims(signal, 1), dtype=tf.float32)

    print_available_ram()
    with beam.Pipeline() as p:
        output = (
            p 
            | beam.Create([("encounter_9182", waveform_exp)])
            | beam.ParDo(InferenceClient("http://127.0.0.1:5000/predict"))
            | beam.ParDo(PostProcess())
        )
    logging.info(output)
    print_available_ram()


if __name__ == "__main__":
    # run()
    simple_run()
        