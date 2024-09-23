import tempfile
import os
import csv

import tensorflow as tf
import numpy as np
import tflite as tfl

# import tensorflow.keras.utils
from tensorflow import keras
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_hub as hub
import itertools
import soundfile as sf
import librosa
import math
import time
# from prune import prune
# import tensorflow_model_optimization as tfmot


interpreter=tf.lite.Interpreter(model_path='data/model/quantized_model1.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

i=0
whales_identified=0
whale_samples=0
other_samples_identified=0
other_samples=0
directory_mn='./clips/mn/'
directory_not_mn='./clips/not_mn/'

# filename = os.fsdecode(file)

# # try opening the file
# waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(directory_mn+filename))

# waveform = tf.slice(tf.squeeze(tf.expand_dims(waveform, 0),[2]),[1,0],[1,15600])  # makes a batch of size 1
waveform = tf.convert_to_tensor(np.random.random((1, 15600)).astype(np.float32))
print(waveform.shape)


# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], waveform)

# Run inference
interpreter.invoke()

# output_details[0]['index'] = the index which provides the input
output = interpreter.get_tensor(output_details[0]['index'])

# If the output type is int8 (quantized model), rescale data
output_type = output_details[0]['dtype']
if output_type == np.int8:
    output_scale, output_zero_point = output_details[0]['quantization']
    print("Raw output scores:", output)
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)
    print()
    output = output_scale * (output.astype(np.float32) - output_zero_point)


print("Output scores:", output)