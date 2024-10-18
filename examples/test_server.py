import requests 
import os


inference_url = os.environ.get("INFERENCE_URL", "0.0.0.0:5000/predict")

data = {
    "key": "encouter1",
    "batch": [
        [0.3745401203632355], [0.9507142901420593], [0.7319939136505127], [0.5986585021018982],
    ]*10_000, 
}

response = requests.post(inference_url, json=data)
print(response.json())
