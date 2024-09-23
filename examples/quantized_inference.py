import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, DirectOptions, GoogleCloudOptions, StandardOptions
from apache_beam.utils.shared import Shared

from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor
from apache_beam.ml.transforms.embeddings.tensorflow_hub import _TensorflowHubModelHandler
from apache_beam.ml.inference.base import RunInference

from typing import Dict, Any, Optional, Callable

import logging
import math
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

import psutil


logging.getLogger().setLevel(logging.INFO)


def print_available_ram():
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert from bytes to GB
    total_ram = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Total RAM: {total_ram:.2f} GB")

print_available_ram()


class QuantizedModelHandler(TFModelHandlerTensor):
    def __init__(
        self,
        model_uri: str,
        inference_fn: Optional[Callable] = None,
        inference_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(model_uri=model_uri, inference_fn=inference_fn, **kwargs)
        self.inference_args = inference_args
        self.batch_size = 15600  # model expects exactly 15600 samples

    def load_model(self):
        interpreter=tf.lite.Interpreter(model_path='data/model/quantized_model1.tflite')
        interpreter.allocate_tensors()
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        print(f"   input_details = {self.input_details}")
        print(f"   output_details = {self.output_details}")

        return interpreter
    
    def run_inference(self, element, model, inference_args, model_id=None):
        print("inside run_inference")

        if type(element) == list:
            if len(element) != 1:
                raise ValueError(f"Expected a single element in batch, but got {len(element)} elements")
            element = np.array(element[0], dtype=np.float32)

        print(f"   element.shape = {element.shape}")
        print(f"   model = {model}")
        print(f"   inference_args = {inference_args}")


        batched_element = quantized_preprocess(element, self.batch_size)

        return self._inference_fn(model, batched_element, self.input_details, self.output_details, model_id)


def quantized_preprocess(signal, batch_size):
    # audio = tf.squeeze(audio, axis=0)
    # audio = tf.slice(audio, [1, 0], [1, 15600])

    logging.info(f"Batching signal {signal.shape} to fit model input size {batch_size}.")
    padding = np.zeros(batch_size - (signal.shape[0] % batch_size))
    padded_signal = np.concatenate([signal, padding])

    split_indices = [batch_size*(i) for i  in range(1, math.floor(padded_signal.shape[0] / batch_size))]
    signal_batches = np.array_split(padded_signal, split_indices)

    # expand dims
    signal_batches = [np.expand_dims(batch, axis=0) for batch in signal_batches]
    breakpoint()
    
    logging.info(f"Split signal into {len(signal_batches)} batches of size {batch_size}.")
    logging.info(f"Size of final batch {signal_batches[0].shape}")

    return signal_batches


def quantized_inference_fn(
    model,
    batches: tf.Tensor,
    input_details,
    output_details,
    model_id: Optional[str] = None,
):
    print_available_ram()

    print("inside quantized_inference_fn")

    outputs = []

    for batch in batches:
        print(f"   batch.shape = {batch.shape}")
        print(f"   batch.dtype = {batch.dtype}")

        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)

        model.set_tensor(input_details[0]['index'], batch)

        # Run inference
        model.invoke()

        # output_details[0]['index'] = the index which provides the input
        print("    waveform = ", batch)
        output = model.get_tensor(output_details[0]['index'])
        print(f"   output = {output}")
        outputs.append(output)

    return output


def simple_run():
    np.random.seed(42)

    # examples = np.random.random((1, 15600)).astype(np.float32)
    examples = np.random.random((16_000*30)).astype(np.float32)  # 30 seconds of random "audio"
    dummy_data = tf.expand_dims(examples, 0) # if we create single dim batch, beam tries to create as many workers as len(exmaples)
    print(f"   dummy_data.shape = {dummy_data.shape}") # (1, 15600)

    # model_handler = _TensorflowHubModelHandler(
    model_handler = QuantizedModelHandler(
        "https://tfhub.dev/google/humpback_whale/1",
        # "https://www.kaggle.com/models/google/humpback-whale/TensorFlow2/humpback-whale/1",
        # "https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2",
        # "/Users/per.morten.halvorsen@schibsted.com/.cache/kagglehub/models/google/humpback-whale/tensorFlow2/humpback-whale/1",
        inference_fn=quantized_inference_fn,
        inference_args={"context_step_samples": tf.cast(10_000, tf.int32)},
        
    )
    
    print_available_ram()
    with beam.Pipeline() as p:
        output = (
            p 
                | beam.Create(dummy_data)
                | RunInference(model_handler) #, inference_args={"context_step_samples": tf.cast(10_000, tf.int64)})
        )
    print(output)
    print_available_ram()


if __name__ == "__main__":
    # run()
    simple_run()
        