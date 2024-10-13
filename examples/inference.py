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

import psutil


logging.getLogger().setLevel(logging.INFO)


def print_available_ram():
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert from bytes to GB
    total_ram = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Total RAM: {total_ram:.2f} GB")

print_available_ram()

# class DummyData(beam.DoFn):
#     name = "DummyData"

#     def process(self, element):
#         np.random.seed(42)
#         dummy = np.random.random((1, 39124, 1)).astype(np.float32)
#         return dummy


# class GoogleHumpbackWhaleInferenceDoFn(beam.DoFn):
#     name = "GoogleHumpbackWhaleInferenceDoFn"

#     def __init__(self):
#         self.model_sample_rate =10000
        
#     def setup(self):
#         # Load the model once per worker
#         self.model = hub.load("https://tfhub.dev/google/humpback_whale/1")
#         self.score_fn = self.model.signatures["score"]
#         logging.info(f"Model loaded inside {self.name}")
    
#     def process(self, element):
#         logging.info(f"Processing element inside {self.name}")
#         np.random.seed(42)
#         dummy = np.random.random((1, 39124, 1)).astype(np.float32)
#         results = self.model(dummy, True, None)
#         logging.info(f"   results.shape = {results.shape}")


#     def _process(self, element):
#         key = "9182"
#         path = element.get(key)

#         signal = np.load(path)

#         # convert to float32
#         signal = signal.astype(np.float32)

#         logging.info(f"   signal.shape = {len(signal)}")
#         logging.info(f"   signal.dtype = {type(signal)}")

#         # Reshape signal
#         logging.info(f"   inital input: len(signal) = {len(signal)}")
#         waveform1 = np.expand_dims(signal, axis=1)
#         waveform_exp = tf.expand_dims(waveform1, 0)
#         logging.info(f"   final input: waveform_exp.shape = {waveform_exp.shape}")
#         logging.info(f"   final input: waveform_exp.dtype = {waveform_exp.dtype}")
#         logging.info(f"   final input: type(waveform_exp) = {type(waveform_exp)}")

#         print("before inference")
#         print_available_ram()

#         # results = self.score_fn(
#         #     waveform=waveform_exp,
#         #     context_step_samples=int(self.model_sample_rate)
#         # )["scores"]
#         results = self.model.score(
#             waveform=waveform_exp,
#             context_step_samples=int(self.model_sample_rate)
#         )["scores"]

#         print("after inference")
#         print_available_ram()

#         # results = waveform_exp
#         logging.info(f"   results.shape = {results.shape}")
        
#         # yield (key, results.tolist())
#         return key


# def run():
#     # Initialize pipeline options
#     pipeline_options = PipelineOptions()
#     # pipeline_options.view_as(SetupOptions).save_main_session = True

#     args = {
#         "9182": "data/audio/butterworth/2016/12/20161221T004930-005030-9182.npy"
#     }

#     with beam.Pipeline(options=pipeline_options) as p:
#         # input_data  = p           | "Create Input"    >> beam.Create([args])  
#         input_data  = p           | "Create Input"    >> beam.ParDo(DummyData())  
#         output_data = input_data  | "Classify Audio"  >> beam.ParDo(GoogleHumpbackWhaleInferenceDoFn())

#     logging.info("Pipeline executed successfully")


class MyTensorflowHubModelHandler(TFModelHandlerTensor):
    def __init__(
        self,
        model_uri: str,
        inference_fn: Optional[Callable] = None,
        inference_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(model_uri=model_uri, inference_fn=inference_fn, **kwargs)
        self.inference_args = inference_args

    def load_model(self):
        model = hub.load(self._model_uri, **self._load_model_args)
        return model

    def run_inference(self, batch, model, inference_args, model_id=None):
        print("inside run_inference")
        print(f"   batch.shape      = {batch}")
        print(f"   model            = {model}")
        print(f"   inference_args   = {inference_args}")

        # serialize 
        batch = [batch[0].numpy().tolist()]
        return self._inference_fn(model, batch, self.inference_args, model_id)


def tensor_inference_fn(
    model: tf.Module,
    batch: tf.Tensor,
    inference_args: Dict[str, Any],
    model_id: Optional[str] = None,
):
  print_available_ram()
  return model.score(waveform=batch, **inference_args)


def simple_run():
    np.random.seed(42)

    examples = np.random.random((39124, 1)).astype(np.float32)
    dummy_data = tf.expand_dims(examples, 0)
    print(f"   dummy_data.shape = {dummy_data.shape}")

    # model_handler = _TensorflowHubModelHandler(
    model_handler = MyTensorflowHubModelHandler(
        "https://tfhub.dev/google/humpback_whale/1",
        # "https://www.kaggle.com/models/google/humpback-whale/TensorFlow2/humpback-whale/1",
        # "https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2",
        # "/Users/per.morten.halvorsen@schibsted.com/.cache/kagglehub/models/google/humpback-whale/tensorFlow2/humpback-whale/1",
        inference_fn=tensor_inference_fn,
        inference_args={"context_step_samples": tf.cast(10_000, tf.int64)},
        # large_model=True,
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
        