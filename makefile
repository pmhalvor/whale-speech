VERSION := 0.0.0
GIT_SHA := $(shell git rev-parse --short HEAD)
PIPELINE_IMAGE_NAME := whale-speech/pipeline:$(VERSION)-$(GIT_SHA)
MODEL_SERVER_IMAGE_NAME := whale-speech/model-server:$(VERSION)-$(GIT_SHA)
PIPELINE_WORKER_IMAGE_NAME := whale-speech/pipeline-worker:$(VERSION)-$(GIT_SHA)
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

build-model-server:
	docker build -t $(MODEL_SERVER_IMAGE_NAME) --platform linux/amd64 -f Dockerfile.model-server .

push-model-server:
	docker tag $(MODEL_SERVER_IMAGE_NAME) $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME)

build-pipeline-worker:
	docker build -t $(PIPELINE_WORKER_IMAGE_NAME) --platform linux/amd64 -f Dockerfile.pipeline-worker .

push-pipeline-worker:
	docker tag $(PIPELINE_WORKER_IMAGE_NAME) $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME)

build-push-pipeline-worker: build-pipeline-worker push-pipeline-worker

test-server:
	python3 examples/test_server.py
