VERSION := 1.0.0
GIT_SHA := $(shell git rev-parse --short HEAD)
TAG := $(VERSION)-$(GIT_SHA)
PIPELINE_IMAGE_NAME := whale-speech/pipeline
MODEL_SERVER_IMAGE_NAME := whale-speech/model-server
PIPELINE_WORKER_IMAGE_NAME := whale-speech/pipeline-worker

PUBLIC_MODEL_SERVER_IMAGE_NAME := $(shell echo $(MODEL_SERVER_IMAGE_NAME) | sed 's/\//-/g')
PUBLIC_PIPELINE_WORKER_IMAGE_NAME := $(shell echo $(PIPELINE_WORKER_IMAGE_NAME) | sed 's/\//-/g')

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
	conda create -n whale-speech python=3.11
	conda activate whale-speech
	sudo apt-get update
	sudo apt-get install libhdf5-dev libsndfile1 gcc
	python3 -m pip install -r requirements/requirements.txt
	python3 -m pip install -r requirements/model-requirements.txt


# Docker related
check-uncommited:
	git diff-index --quiet HEAD

build: check-uncommited
	docker build -t $(PIPELINE_IMAGE_NAME):$(TAG) --platform linux/amd64 .

push: check-uncommited
	docker tag $(PIPELINE_IMAGE_NAME):$(TAG) $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME):$(TAG)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_IMAGE_NAME):$(TAG)

build-push: build push


# Model server related
build-model-server: check-uncommited
	docker build -t $(MODEL_SERVER_IMAGE_NAME):$(TAG) --platform linux/amd64 -f Dockerfile.model-server .

push-model-server: check-uncommited
	docker tag $(MODEL_SERVER_IMAGE_NAME):$(TAG) $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME):$(TAG)
	docker push $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME):$(TAG)

push-model-server-latest: check-uncommited
	docker tag $(MODEL_SERVER_IMAGE_NAME):$(TAG) $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME):latest
	docker push $(MODEL_REGISTERY)/$(MODEL_SERVER_IMAGE_NAME):latest

build-push-model-server: build-model-server push-model-server push-model-server-latest

publish-latest-model-server: build-model-server
	docker tag $(MODEL_SERVER_IMAGE_NAME):$(TAG) $(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_MODEL_SERVER_IMAGE_NAME):latest
	docker push $(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_MODEL_SERVER_IMAGE_NAME):latest

test-server:
	python3 examples/test_server.py


# Pipeline worker related
build-pipeline-worker: check-uncommited
	docker build -t $(PIPELINE_WORKER_IMAGE_NAME):$(TAG) --platform linux/amd64 -f Dockerfile.pipeline-worker .

push-pipeline-worker: check-uncommited
	docker tag $(PIPELINE_WORKER_IMAGE_NAME):$(TAG) $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME):$(TAG)
	docker push $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME):$(TAG)

push-pipeline-worker-latest: check-uncommited
	docker tag $(PIPELINE_WORKER_IMAGE_NAME):$(TAG) $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME):latest
	docker push $(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME):latest

build-push-pipeline-worker: build-pipeline-worker push-pipeline-worker push-pipeline-worker-latest

publish-latest-pipeline-worker: build-pipeline-worker
	docker tag $(PIPELINE_WORKER_IMAGE_NAME):$(TAG) $(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_PIPELINE_WORKER_IMAGE_NAME):latest
	docker push $(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_PIPELINE_WORKER_IMAGE_NAME):latest


# Pipeline run related
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
		--worker_harness_container_image=$(MODEL_REGISTERY)/$(PIPELINE_WORKER_IMAGE_NAME):latest \
		--start "2024-09-09" \
		--end "2024-09-12" \
		--offset 0 \
		--margin 1800 \
		--batch_duration 60 
# --worker_harness_container_image=$(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_PIPELINE_WORKER_IMAGE_NAME):latest \

run-direct:
	python3 src/pipeline.py \
		--job_name "whale-speech-$(GIT_SHA)" \
		--filesystem gcp \
		--inference_url $(INFERENCE_URL) \
		--runner DirectRunner \
		--worker_harness_container_image=$(PUBLIC_MODEL_REGISTERY)/$(PUBLIC_PIPELINE_WORKER_IMAGE_NAME) \
		--start "2024-09-09" \
		--end "2024-09-12" \
		--offset 0 \
		--margin 1800 \
		--batch_duration 60 


rebuild-run-dataflow: build-push-pipeline-worker run-dataflow

rebuild-run-direct: build-push-pipeline-worker run-direct

show-url:
	echo $(INFERENCE_URL)