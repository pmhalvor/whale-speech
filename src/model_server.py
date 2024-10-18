from flask import Flask, request, jsonify
import tensorflow_hub as hub
import os 
import numpy as np
import tensorflow as tf

import logging

from config import load_pipeline_config
config = load_pipeline_config()

# Enable verbose logging  
logger = logging.getLogger("model_server")
logger.setLevel(logging.INFO)

# Load the TensorFlow model
logger.info("Loading model...")
model = hub.load(config.classify.model_uri)
score_fn = model.signatures["score"]
logger.info("Model loaded.")

# Initialize Flask app
app = Flask("model_server")

# Define inference endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the request data
        data = request.json
        batch = np.array(data['batch'], dtype=np.float32)  # Assuming batch is passed as a list
        key = data['key']
        logger.info(f"batch.shape = {batch.shape}")
        
        # Prepare the input for the model
        waveform_exp = tf.expand_dims(batch, 0)  # Expanding dimensions to fit model input shape
        logger.debug(f"waveform_exp.shape = {waveform_exp.shape}")

        # Run inference
        results = score_fn(
            waveform=waveform_exp, # waveform_exp,
            context_step_samples=config.classify.model_sample_rate
        )["scores"][0] # NOTE currently only support batch size 1
        logger.info(f"results.shape = {results.shape}")
        logger.debug("results = ", results)

        # Return the predictions and key as JSON
        return jsonify({
            'key': key,
            'predictions': results.numpy().tolist()
        })
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == "__main__":
    logger.info(f"Host: {config.general.host} port: {config.general.port}")

    port = os.environ.get('PORT', 8080)

    app.run(host=config.general.host, port=port, debug=config.general.debug)
