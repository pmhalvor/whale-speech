VERSION := 0.0.0
GIT_SHA := $(shell git rev-parse --short HEAD)
PIPELINE_IMAGE_NAME := whale-speech/pipeline:$(VERSION)-$(GIT_SHA)
MODEL_SERVER_IMAGE_NAME := whale-speech/model-server:$(VERSION)-$(GIT_SHA)
PIPELINE_WORKER_IMAGE_NAME := whale-speech/pipeline-worker:$(VERSION)-$(GIT_SHA)
ENV_LOCATION := .env

local-run: 
	bash scripts/kill_model_server.sh
	python3 src/model_server.py & python3 src/pipeline.py
	bash scripts/kill_model_server.sh

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

check-uncommited:
	git diff-index --quiet HEAD

build: check-uncommited
	docker build -t $(PIPELINE_IMAGE_NAME) --platform linux/amd64 .

push: check-uncommited
	docker tag $(PIPELINE_IMAGE_NAME) $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME)

build-push: build push

build-model-server: check-uncommited
	docker build -t $(MODEL_SERVER_IMAGE_NAME) --platform linux/amd64 -f Dockerfile.model-server .

push-model-server: check-uncommited
	docker tag $(MODEL_SERVER_IMAGE_NAME) $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME)

build-push-model-server: build-model-server push-model-server

build-pipeline-worker: check-uncommited
	docker build -t $(PIPELINE_WORKER_IMAGE_NAME) --platform linux/amd64 -f Dockerfile.pipeline-worker .

push-pipeline-worker: check-uncommited
	docker tag $(PIPELINE_WORKER_IMAGE_NAME) $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME)

build-push-pipeline-worker: build-pipeline-worker push-pipeline-worker

test-server:
	python3 examples/test_server.py

run-dataflow:
	python3 src/pipeline.py \
		--job_name "whale-speech-$(GIT_SHA)" \
		--filesystem gcp \
		--inference_url $(INFERENCE_URL) \
		--runner DataflowRunner \
		--region us-central1 \
		--worker_machine_type=n1-highmem-8 \
		--disk_size_gb=100 \
		--num_workers=8 \
		--max_num_workers=8 \
		--autoscaling_algorithm=THROUGHPUT_BASED \
		--worker_harness_container_image=$(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME) \
		--start "2024-07-11" \
		--end "2024-07-11" \
		--offset 0 \
		--margin 1800 \
		--batch_duration 60 

run-direct:
	python3 src/pipeline.py \
		--job_name "whale-speech-$(GIT_SHA)" \
		--filesystem gcp \
		--inference_url $(INFERENCE_URL) \
		--runner DirectRunner \
		--worker_harness_container_image=$(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME) \
		--start "2024-07-11" \
		--end "2024-07-11" \
		--offset 0 \
		--margin 600


rebuild-run-dataflow: build-push-pipeline-worker run-dataflow

rebuild-run-direct: build-push-pipeline-worker run-direct

show-url:
	echo $(INFERENCE_URL)