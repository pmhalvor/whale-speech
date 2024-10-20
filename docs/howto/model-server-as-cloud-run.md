# Model server as Cloud Run
In this guide, we will deploy the model server as a [Cloud Run](https://cloud.google.com/run/) service. 

Cloud Run is a serverless compute platform that allows you to run prebuilt containers triggered via HTTP requests.
Our model server component is a perfect example of a service that can be deployed on Cloud Run, since it is a REST API listening for POST requests on a specified port and endpoint.

## Prerequisites
- A Google Cloud Platform (GCP) account and [project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) with [billing enabled](https://cloud.google.com/billing/docs/how-to/modify-project).
- [Docker](https://github.com/docker/docker-install?tab=readme-ov-file#usage) installed on your local machine.
- This code locally cloned (`git clone https://github.com/pmhalvor/whale-speech`).

## Steps

### 0. (Optional) Set up Artifact Registry
If you want to store your Docker images in Google Cloud, you can use [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview).

You'll likely need to enable the app, create a repository, then add permissions to your local environment to push to the repository.
See more on this authentication process [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#auth).

### 1. Build the Docker image and push to Google Container Registry
Navigate to the project directory in a terminal, build and tag your model-server image, and push to your model registry.

If you are using the Google Artifact Registry, you'll need to tag your image with the registry URL and a zone, something like `us-central1-docker.pkg.dev/your_project/whale-speech/model-server:x.y.z`.
If you prefer to use the free Docker hub registry, you can use your public Docker ID as a prefix to your image tag, something like `your_docker_id/whale-speech:model-server-x.y.z`.

This guide will only document the Google Artifact Registry method. The Docker Hub method is similar, though naming might be different. 

```bash
cd whale-speech
docker build -f Dockerfile.model-server -t us-central1-docker.pkg.dev/your_project/whale-speech/model-server:x.y.z  .
docker push us-central1-docker.pkg.dev/your_project/whale-speech/model-server:x.y.z
```

The `Dockerfile.model-server` is a Dockerfile written for hosting the model server. 
You can find this file in the `whale-speech` directory.
Note there is no need to expose a port in the Dockerfile, as this will be done in the Cloud Run deployment.


### 2. Deploy image as Cloud Run service
Navigate to the [Cloud Run](https://console.cloud.google.com/run) page in the GCP console.

- Select **Deploy container** and then **Service**, since we'll want the container to be server with an endpoint.
- Add you container image URL that you pushed to in the step above `docker push us-central1-docker.pkg.dev/your_project/whale-speech/model-server:x.y.z`.
- Name your service (ex. `whale-speech-model-server`) and select a region (`us-central1` is a good default).
- Open the section for **Container(s), Volumes, Networking, Security**.
    - Add the port your model server is listening on (default is `5000`) as the container port. This will be added as an environment variable when running the container.
    - Update memory and CPU count as needed. I noticed that 4 GiB and 2 vCPUs worked fine with batch durations of 60 seconds. This value can be adjusted through revisioning later. 
    - I'd maybe reccomend lowering the max number of requests per container to 1-5, since the inputs will be larger for each request.
    - You may need to adjust the min and max number of instances, depending on your expected traffic and quotas. 

- Click **Create**.

### 3. Test the service
Once the service is deployed, you can test it by sending a POST request to the service's endpoint.
The URL should be available at to top of the service details page. It'll look something like `https://whale-speech-model-server-xxxx.a.run.app`.

In the `whale-speech` directory, you can run the following command to test the service:
```bash
export INFERENCE_URL="https://whale-speech-model-server-xxxx.a.run.app"
python3 examples/test_model_server.py
```

The expected response should be a JSON object with a `prediction` key and a list of floats as the value.

I'd recommend saving the `export INFERENCE_URL="https://whale-speech-model-server-xxxx.a.run.app"` command to an `env.sh` file in the `whale-speech` directory, so you can easily run the test script in the future. This filename is in the `.gitignore`, so it won't be pushed to the repository.

In the same file, I export a MODEL_REGISTRY variable, which is the URL of the model server image in the Google Artifact Registry. This is used in the `make` targets, like `build-model-server`, which builds the image to the registry.


### Trouble shooting
If you are having trouble with the deployment, you can check the logs in the Cloud Run console.

Some common issues I've run into are:
- Not exposing the correct port in the container settings.
- Too low memory or CPU count.
- IAM permissions between Artifact Registry and Cloud Run. Read more [here](https://cloud.google.com/artifact-registry/docs/access-control#iam).