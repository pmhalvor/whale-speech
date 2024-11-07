# Utilizing Volume Mounts

During stress-testing the pipeline, we fine some bottlenecks particularly in the I/O operations.
To mitigate this, we can utilize volume mounts to speed up the process.

We should be careful about where we choose to use volumes though, since persisting data to buckets from parallel workers could result in corrupted data, for example if workers write to the same file at the same time. 
That should really only be a problem if you are writing to the same file, which should not happen in this pipeline.

This document aims to decide where to use volume mounts, how to implement and configure them.

# 1. Where to use volume mounts
Our pipeline consists of two main parts: the **model server** and the **pipeline** itself.
The model server is a simple Flask server that serves the model, and the pipeline is a Dataflow job that processes the data.

## Model server
Hosted as its own Cloud Run service on GCP (or a parallel process locally), the most important part of the model server is to classify input audio. 
Ideally, this component should live independent of the pipeline, so that future models can be easily swapped in and out without affecting the pipeline.

### Options
#### A. _Recieve inputs through REST API (POST requests)_
This is the current implementation. 

**Pros**
- Completely isolated from the pipeline. Could be its own service. Acts like a third-party API. 
- Easy to monitor costs, usage, and logs separately from the pipeline.
- Could be exposed as external service on its own. Potentially beneficial if pipeline ends up being not so useful. 

**Cons**
- Requires large network I/O for every request. Could be slow if many requests are made.
- Limits on size of requests. Forced to manually batching and sending requests, causing many more requests per run. Due to parallelism, this could mean many requests simultaneously, which the Cloud Run service will complain about and potentially block.

#### B. _Load inputs through volume mounts_
Pipeline sends POST requests with paths to audio within the volume, instead of the data itself. 
Whether the model returns the classification array, or writes to volume and returns paths is to be decided.

**Pros**
- Much smaller I/O sizes. Only need to send the path to the data, not the data itself.
- No limits on size of requests. Can send all data at once. This would require the model server to smartly batch inputs if sizes are too large. This is not necessarily a bad thing, since it would be easier to ensure the correct order of batched model outputs. 
- Removes chances of model server getting DDOSed by too many requests at once from the pipeline. 
- Could potentially be faster, since the model server can read from the volume directly.
- Might better leverage the stateless nature of the model server, by scaling up to handle new requests on different instances. 

**Cons**
- Compute resources hosting the model server need to be scalable to handle large data and model weights. Currently blocked by scaling quotas in Google Cloud Run, but theoretically, this should not be a problem.
- Requires the model server to have access to the volume. This could be a security risk, since the model server could potentially read any data in the volume. Though, with the data being public, this is not a huge concern.
- Requires all preprocessing to have been complete in the previous steps. This means implementation now will require a rewrite of [sift stage](../../src/stages/sift.py) (should be changed to "preprocess audio", where sift-parsing and resampling are done in the same step).

## Pipeline
The main pipeline is a Beam workflow aimed to run on a Dataflow runner, meaning it is a distributed system.
Each stage is currently implemented to write its outputs to Google Cloud Storage, while passin gthe same data along to the next stage.
The pipeline should be able to run locally as well, for development and low-resource runs. 

Using volumes here can be useful for persistent storage, but we don't want to sacrifice the parallelism of the pipeline.

### Options
#### A. _Write to GCS using Beam's I/O connectors_
This is the current implementation, which does the job well.

**Pros**
- Already implemented.
- Explicit full paths (or path templates) passed in configs make logging and debugging more understandable. Full paths are written to, logged, then passed to final output tables. 

**Cons**
- Feels less orgainzed or robust. Every write is its own operation, which requires connecting to GCS, writing, and closing the connection. This could be slow if many writes are done.
- Could be more expensive, since every write is a separate operation. (I'm not really sure about this, but could be worth considering.)

#### B. _Mount GCS bucket as a volume_
The bucket where our outputs are written to is mounted to the container running the pipeline.

**Pros**
- Faster writes. Since the bucket is mounted, writes are done directly to the bucket. This could be faster than connecting to GCS for every write.
- Feels like the "true MLOps" choice. Getting the most out of Docker functionality. This may be overkill for this project, but would serve as a good exaample for future projects where this may be even more useful.

**Cons**
- Going to take a while to implement this for all stages. We just finished implementing separate writes per stage run.
- Not sure how this will change between running with Dataflow and running locally. Will need to test this.
  - We only provide the image as a `worker_harness_container_image`. Mounting requires adding the volume tag when running the image. Would need to check that Beam supports this.

# 2. How to implement volume mounts
## Model server
- What will internal paths to data look like? How does this vary from bucket strcture versus local structure? How detailed of a path can we provide for the mounted volume?
  - Currently looks like we can only provide the top level bucket (`bioacoustics`). This would mean the path inside the app would need to be `whale_speech/data/audio`
  - Alternatively, we can change the name of the bucket to `whale_speech`, since its already in the bioacoustics project. This would mean the path inside the app would be `data/audio/...` similar to local.
- Should we continue to support recieving data through REST API? Most likely, yes, to avoid breaking backward compatibility.
  - If so, how do we check when path is provided versus data? 
- Where do model configs lie? 
  - How to define different batch sizes? 
  - Is everything just baked into the image from before or shuold we have multiple 

## Pipeline
- How to mount the bucket to the container?
  - How does this change between running locally and running on Dataflow?
- How to write to the mounted volume?
- Is this even worth it?

# Decision

## Model server
We will implement the volume mounts for the model server.
We will continue to support recieving data through REST API, but will also support recieving paths to data.
We will mount the bucket to the container, and write to the mounted volume.
The paths to data will be the same as the local paths, menaing the top level bucket will now be called `whale_speech`.

## Pipeline
We will not implement volume mounts for the pipeline.
We will continue to write to GCS using Beam's I/O connectors.

