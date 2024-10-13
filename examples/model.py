import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

import psutil

np.random.seed(42)

def print_available_ram():
    memory_info = psutil.virtual_memory()
    available_ram = memory_info.available / (1024 ** 3)  # Convert from bytes to GB
    total_ram = memory_info.total / (1024 ** 3)  # Convert from bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Total RAM: {total_ram:.2f} GB")

print_available_ram()


# model = hub.load("https://tfhub.dev/google/humpback_whale/1")
model = hub.load("https://www.kaggle.com/models/google/humpback-whale/TensorFlow2/humpback-whale/1")
score_fn = model.signatures["score"]
# print(model.__dict__.keys())
print(model.tensorflow_version)
print(model.graph_debug_info)

# signal = np.load("data/audio/butterworth/2016/12/20161221T004930-005030-9182.npy")

# waveform1 = np.expand_dims(signal, axis=1)
# waveform_exp = tf.expand_dims(waveform1, 0)
# print(f"   final input: waveform_exp.shape = {waveform_exp.shape}")
# print(f"   final input: waveform_exp.dtype = {waveform_exp.dtype}")
# print(f"   final input: type(waveform_exp) = {type(waveform_exp)}")

print_available_ram()

dummy = np.random.random((1, 39124, 1)).astype(np.float32)
print(f"   final input: dummy.shape = {dummy.shape}")
# results = model(dummy, True, None)
results = model.score(
    waveform=dummy, # waveform_exp,
    context_step_samples=10_000
)["scores"]
# # print(model.__dict__.keys())
# # print(model.signatures.keys())

print_available_ram()

# print(waveform_exp.shape)
print("input: ", dummy.shape)
print("result:", results.shape)
print("result:", results)