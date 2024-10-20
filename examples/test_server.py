import requests 
import os


inference_url = os.environ.get("INFERENCE_URL", "0.0.0.0:5000/predict")

data = {
    "key": "encouter1",
    "batch": [
        [0.3745401203632355], [0.9507142901420593], [0.7319939136505127], [0.5986585021018982], [0.15601864044243652], [0.15599452033620265]
    ]*10_000*10, # (6 samples * 10_000 = 6 seconds )* 10 = 60 seconds 
}

response = requests.post(inference_url, json=data)
if response.status_code == 200:
    print(response.json())
else:
    print(response.text)