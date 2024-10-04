from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf

import logging

from config import load_pipeline_config
config = load_pipeline_config()

# Load the TensorFlow model
logging.info("Loading model...")
model = hub.load("https://www.kaggle.com/models/google/humpback-whale/TensorFlow2/humpback-whale/1")
# model = hub.load("https://tfhub.dev/google/humpback_whale/1")
score_fn = model.signatures["score"]
logging.info("Model loaded.")

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
        logging.info(f"batch.shape = {batch.shape}")
        
        # Prepare the input for the model
        waveform_exp = tf.expand_dims(batch, 0)  # Expanding dimensions to fit model input shape
        logging.debug(f"waveform_exp.shape = {waveform_exp.shape}")

        # Run inference
        results = score_fn(
            waveform=waveform_exp, # waveform_exp,
            context_step_samples=config.classify.model_sample_rate
        )["scores"][0] # NOTE currently only support batch size 1
        logging.info(f"results.shape = {results.shape}")
        logging.debug("results = ", results)

        # Return the predictions and key as JSON
        return jsonify({
            'key': key,
            'predictions': results.numpy().tolist()
        })
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
