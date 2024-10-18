local-run: 
	bash scripts/kill_model_server.sh
	python3 src/model_server.py & python3 src/pipeline.py
	bash scripts/kill_model_server.sh
	python3 src/gcp.py --deduplicate

run-pipeline:
	python3 src/pipeline.py

model-server:
	python3 src/model_server.py

kill-model-server:
	bash scripts/kill_model_server.sh

gcp-init:
	python3 src/gcp.py --init
	
gcp-deduplicate:
	python3 src/gcp.py --deduplicate

setup:
	python3 -m venv .env
	.env/bin/pip install --upgrade pip
	.env/bin/pip install -r requirements/requirements.txt
	source .env/bin/activate

activate:
	source .env/bin/activate

run: activate run-pipeline