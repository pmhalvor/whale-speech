import tensorflow as tf
import numpy as np

import tensorflow as tf


interpreter=tf.lite.Interpreter(model_path='data/model/quantized_model1.tflite')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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