# 📣🐋 Whale Speech 
A pipeline to map whale encounters to hydrophone audio.

<sub>
Derived from <a href="https://docs.mbari.org/pacific-sound/notebooks/humpbackwhales/detect/PacificSoundDetectHumpbackSong/"> PacificSoundDetectHumpbackSong</a>, though not directly affiliated with MBARI, NOAA, or HappyWhale.
</sub>


## Pipeline description

Stages:
1. **Input**: When (and where*) to look for whale encounters on [HappyWhale](https://happywhale.com/).
2. **Geometry Search**: Query [open-oceans/happywhale](https://github.com/open-oceans/happywhale) to find potential whale encounters. 

   &rarr; Expected outputs: encounter ids, start and end times, and longitude and latitude.

3. **Retrive Audio**: Download audio from MBARI's [Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/) around the time of the encounter. 
    
    &rarr; Expected outputs: audio array, start and end times, and encounter ids.
    
4. **Parse Audio**: Break audio into non-overlaping segments with flagged frequency detections. 
        
    &rarr; Expected outputs: cut audio array, detection intervals, and encounter ids.

5. **Classify Audio**: Use a NOAA and Google's [humpback_whale model](https://tfhub.dev/google/humpback_whale/1) to classify the flagged segments.

    &rarr; Expected outputs: resampled audio, classification score array, and encounter ids.

6. **Postprocess Labels**: Build clip-intervals for each encounter for playback snippets.

    &rarr; Expected outputs: encounter ids, cut/resampled audio array, and aggregated classification score.

7. **Output**: Map the whale encounter ids to the playback snippets.

<!-- Light mode -->
[![](https://mermaid.ink/img/pako:eNpVkttOwkAQhl9lMleaFIJFTo0x4SBIItGoV1ouhnZKm2y7ZA9oJby7S1uJzNX-s98cMweMZMwYYCLkV5SSMvA-CwtwNv5MKEioFZHgIiYFy2JnjV5Dq3UPk6v6cyvkhmHBMmejSnhjUlF63SSoyGlD5lZnEbw6LNszjG2cyYab1FwtppV4aIKSTBhW8EJKX8Y8VNi8wTZWCM0lw1SQ1llSXrBzuGu3Hb1saCU30sDKzS1cw2rP6gyeki4azNAWXqQ2OyUj1hqeaMNCN-iiQh8__9rUKTxb4_az_j_TAk4KPcxZ5ZTFbs-HkydEk3LOIQbuGXNCVpgQw-LoULJGvpVFhIFRlj1U0m5TdFXchB7aXUyGZxltFeVn746KDykvNAYH_Mag2_HbN0P_ZtjvjUa3frff9bDEoHP08KeK6LRHtQ38nt8b3A4HHnKcGalW9WFU93H8BWH3qDQ?type=png)](https://mermaid.live/edit#pako:eNpVkttOwkAQhl9lMleaFIJFTo0x4SBIItGoV1ouhnZKm2y7ZA9oJby7S1uJzNX-s98cMweMZMwYYCLkV5SSMvA-CwtwNv5MKEioFZHgIiYFy2JnjV5Dq3UPk6v6cyvkhmHBMmejSnhjUlF63SSoyGlD5lZnEbw6LNszjG2cyYab1FwtppV4aIKSTBhW8EJKX8Y8VNi8wTZWCM0lw1SQ1llSXrBzuGu3Hb1saCU30sDKzS1cw2rP6gyeki4azNAWXqQ2OyUj1hqeaMNCN-iiQh8__9rUKTxb4_az_j_TAk4KPcxZ5ZTFbs-HkydEk3LOIQbuGXNCVpgQw-LoULJGvpVFhIFRlj1U0m5TdFXchB7aXUyGZxltFeVn746KDykvNAYH_Mag2_HbN0P_ZtjvjUa3frff9bDEoHP08KeK6LRHtQ38nt8b3A4HHnKcGalW9WFU93H8BWH3qDQ)

<!-- Dark mode -->
<!-- [![](https://mermaid.ink/img/pako:eNpVkttOwkAQhl9lMleaFALl3BgTzpJIJOKVlIttO6WN2y7ZA1oJ7-7SViNztf_sN8fMGUMREXoYc_EZJkxqeJv5OVgb72LmxawRMk55xCSs8qPRag-NxiNM7qrPAxcBwZJERloWsCUmw-S-TlCS05rMjEpDeLVYeiIYmygVNTepuEpMSzGvg-KUa5KwYVLdxsxLbFFjgeFcUUEw5UypNC5u2AU8NJuWXtW0FIHQsLZzc9uwPJH8A69JlzWm2QE2QumjFCEpBc8sIK5qdFmiT7vfNlUCL0bb_ez_z7SEq0IHM5IZSyO75_PV46NOKCMfPfu0q_3w0c8vlmNGi22Rh-hpachBKcwhQVvCjuegOUZM0yxlB8myP--R5e9C3Gj0zviFXqflNttDtz3s90ajrtvpdxws0GtdHPwuI1rNUWUDt-f2Bt3hwEGKUi3kurqK8jguPzO5pvE?type=png)](https://mermaid.live/edit#pako:eNpVkttOwkAQhl9lMleaFALl3BgTzpJIJOKVlIttO6WN2y7ZA1oJ7-7SViNztf_sN8fMGUMREXoYc_EZJkxqeJv5OVgb72LmxawRMk55xCSs8qPRag-NxiNM7qrPAxcBwZJERloWsCUmw-S-TlCS05rMjEpDeLVYeiIYmygVNTepuEpMSzGvg-KUa5KwYVLdxsxLbFFjgeFcUUEw5UypNC5u2AU8NJuWXtW0FIHQsLZzc9uwPJH8A69JlzWm2QE2QumjFCEpBc8sIK5qdFmiT7vfNlUCL0bb_ez_z7SEq0IHM5IZSyO75_PV46NOKCMfPfu0q_3w0c8vlmNGi22Rh-hpachBKcwhQVvCjuegOUZM0yxlB8myP--R5e9C3Gj0zviFXqflNttDtz3s90ajrtvpdxws0GtdHPwuI1rNUWUDt-f2Bt3hwEGKUi3kurqK8jguPzO5pvE) -->




<sub>
*Currently only support encounters around the Monterey Bay Hydrophone (<a href="https://www.mbari.org/technology/monterey-accelerated-research-system-mars/">MARS</a>).
</sub>

<br>

## Getting started

### Install

Create a virtual environment and install the required packages.
We'll use conda for this, but you can use any package manager you prefer.

Since we're developing on an M1 machine, we'll need to specify the `CONDA_SUBDIR` to `osx-arm64`.
This step should be adapted based on the virtual environment you're using.

#### M1:
```bash
CONDA_SUBDIR=osx-arm64 conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```


#### Other:
```bash
conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```

### Google Cloud SDK
To run the pipeline on Google Cloud Dataflow, you'll need to install the Google Cloud SDK.
You can find the installation instructions [here](https://cloud.google.com/sdk/docs/install).

Make sure you authentication your using and initialize the project you are using.
```bash
gcloud auth login
gcloud init
```

For newly created projects, each of the services used will need to be enabled. 
This can be easily done in the console, or via the command line. 
For example:
```bash
gcloud services enable bigquery.googleapis.com
gcloud services enable dataflow.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable run.googleapis.com
```

### Run locally 
To run the pipeline and model server locally, you can use the `make` target `local-run`.

```bash
make local-run
```

This target starts by killing any previous model servers that might be running (needed for when a pipeline fails, without tearing down the server, causing the previous call to hang). 
Then it starts the model server in the background and runs the pipeline.


### Build and push the model server
To build and push the model server to your model registry (stored as an environment variable), you can use the following `make` target.

```bash
make build-push-model-server
```
This target builds the model server image and pushes it to the registry specified in the `env.sh` file.
The tag is a combination of the version set in the makefile and the last git commit hash. 
This helps keep track of what is included in the image, and allows for easy rollback if needed.
The target fails if there are any uncommited changes in the git repository.

The `latest` tag is only added to images deployed via GHA.

### Run pipeline with Dataflow
To run the pipeline on Google Cloud Dataflow, you can use the following `make` target.

```bash
make run-dataflow
```
Logging in the terminal will tell you the status of the pipeline, and you can follow the progress in the [Dataflow console](https://console.cloud.google.com/dataflow/jobs).

In addition to providing the inference url and filesystem to store outputs on, the definition of the above target also provides an example on how a user can pass additional arguments to and request different resources for the pipeline run. 

**Pipeline specific parameters**
You can configure all the paramters set in the config files directly when running the pipeline.
The most important here is probably the start and end time for the initial search. 

```bash
		--start "2024-07-11" \
		--end "2024-07-11" \
		--offset 0 \
		--margin 1800 \
		--batch_duration 60 
```

Note that any parameters with the same name under different sections will only be updated if its the last section in the list. 
Also, since these argparse-parameters are added automatically, behavior of boolean flags might be unexpected (always true is added). 
<!-- TODO fix behavior of boolean in-line parameters -->

**Compute resources**
The default compute resources are quite small and slow. To speed things up, you can request more workers and a larger machine type. For more on Dataflow resources, check out [the docs](https://cloud.google.com/dataflow/docs/reference/pipeline-options#worker-level_options).
```
		--worker_machine_type=n1-highmem-8 \
		--disk_size_gb=100 \
		--num_workers=8 \
		--max_num_workers=8 \
```


Note, you may need to configure IAM permissions to allow Dataflow Runners to access images in your Artifact Registry. Read more about that [here](https://cloud.google.com/dataflow/docs/concepts/security-and-permissions).


## Resources 
- [HappyWhale](https://happywhale.com/)
- [open-oceans/happywhale](https://github.com/open-oceans/happywhale)
- [MBARI's Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/)
- [NOAA and Google's humpback_whale model](https://tfhub.dev/google/humpback_whale/1)
- [Monterey Bay Hydrophone MARS](https://www.mbari.org/technology/monterey-accelerated-research-system-mars/)
- [Google Cloud Console](https://console.cloud.google.com/)
