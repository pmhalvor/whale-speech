GIT_SHA := $(shell git rev-parse --short HEAD)
PIPELINE_IMAGE_NAME := whale-speech/pipeline:$(GIT_SHA)
MODEL_REGISTERY := us-central1-docker.pkg.dev/bioacoustics-2024
ENV_LOCATION := .env

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
	apt-get update
	apt-get install python3-venv libhdf5-dev libsndfile1 gcc
	python3 -m venv $(ENV_LOCATION)
	$(ENV_LOCATION)/bin/pip install -r requirements/requirements.txt

run: 
	$(ENV_LOCATION)/bin/python3 src/pipeline.py

build:
	docker build -t $(PIPELINE_IMAGE_NAME) .

push:
	docker tag $(PIPELINE_IMAGE_NAME) $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME)

deploy:
	gcloud run deploy pipeline --image $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME) --platform managed --region us-central1 --allow-unauthenticated