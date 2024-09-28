from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf

import logging


# Load the TensorFlow model
print("Loading model...")
# model = hub.load("https://tfhub.dev/google/humpback_whale/1")
model = hub.load("https://www.kaggle.com/models/google/humpback-whale/TensorFlow2/humpback-whale/1")
score_fn = model.signatures["score"]
print("Model loaded.")

# Initialize Flask app
app = Flask(__name__)

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the request data
        data = request.json
        batch = np.array(data['batch'], dtype=np.float32)  # Assuming batch is passed as a list
        key = data['key']
        print(f"batch.shape = {batch.shape}")
        
        # Prepare the input for the model
        waveform_exp = tf.expand_dims(batch, 0)  # Expanding dimensions to fit model input shape
        print(f"waveform_exp.shape = {waveform_exp.shape}")

        # Run inference
        results = score_fn(
            waveform=waveform_exp, # waveform_exp,
            context_step_samples=10_000
        )["scores"]
        print(f"results.shape = {results.shape}")
        print("results = ", results)

        # Return the predictions and key as JSON
        return jsonify({
            'key': key,
            'predictions': results.numpy().tolist()
        })
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
