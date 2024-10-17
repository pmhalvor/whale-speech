import requests 
import argparse


parser = argparse.ArgumentParser(description='Load config and override with command-line arguments.')
parser.add_argument('--inference_url', type=str, default='0.0.0.0', required=False)
parser.add_argument('--port', type=str, default='5000', required=False)

args = parser.parse_args()

inference_url = args.inference_url
port = args.port

data = {
    "key": "encouter1",
    "batch": [
        [0.3745401203632355], [0.9507142901420593], [0.7319939136505127], [0.5986585021018982],
    ]*10_000, 
}

response = requests.post(f'http://{inference_url}:{port}/predict', json=data)
print(response.json())
