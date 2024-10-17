# How to spin up model server

Our pipeline requires a publically accessible model server for classifications. 
While this code is included in this repo, users will still need to spin up their own server.
This doc will be a walk-through for how I achieved this in GCP.

## Pre-requisites
- A [GCP project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) with [billing enabled](https://cloud.google.com/billing/docs/how-to/modify-project).
- [`gcloud`](https://cloud.google.com/sdk/gcloud) installed and authenticated with your project (`gcloud init`).
- (Optional) This code locally cloned (`git clone https://github.com/pmhalvor/whale-speech`).
- (Optional) [Docker](https://github.com/docker/docker-install?tab=readme-ov-file#usage) installed on your local machine, and authenticated with your [Docker ID](https://docs.docker.com/accounts/create-account/).


## Steps

### 0. (Optional) Build the Docker image and push to Docker Hub
_This step only needs to be done if you are changing model server code or ports._ 
_You can skip this step, and rather use the pre-built image on Docker Hub:_ 
[whale-speech:model-server-X.Y.Z](https://hub.docker.com/r/permortenhalvorsen024/whale-speech/tags). 

Navigate to the project directory, build and tag your model-server Docker image, and push to Docker Hub.

When building the image, you'll need to use your public Docker ID as a prefix to your image tag.
This will tell Docker where to push the image now, and where to pull from later.
Image hosting is free, as long as you make your images public.
Check your team's requirements on whether or not you can make images public or not. 


```bash
cd whale-speech
docker build -f Dockerfile.model-server -t your_docker_id/whale-speech:model-server-x.y.z  .
docker push your_docker_id/whale-speech:model-server-x.y.z
```

### 1. Create Compute Engine instance
- Navigate to the [Compute Engine](https://console.cloud.google.com/compute/instances) page in the GCP console.
- Ensure you are in the correct project.
- Select **Create Instance**.
- Name your instance, select a region and zone, and machine type. If you are using the default image from Step 0, you'll need to make sure to use an `arm64` machine type and update your selected **Boot disk** to an `arm64` based image.
- Under **Firewall**, check the box for **Allow HTTP traffic**.
- Under **Advanced options** > **Networking**, add a new **Network tag**. This will be used to create a firewall rule in the next step. I'll call this tag `whale-speech-inference`, but you can call it whatever you like.
- Click **Create**.

If all goes well, your instance should be up and running in a few minutes.

### 2. Create Firewall Rule
_This section follows the instructions originally posted by Carlos Rojas on [StackOverflow](https://stackoverflow.com/a/21068402/11260232)_.

While waiting for the instance to spin up, we'll update the firewall to allow reading through our specified port. 

- Navigate to [Networking > VPC Network > Firewalls](https://console.cloud.google.com/networking/firewalls) in the GCP console.
- Click **Create Firewall Rule**.
- Name your rule something descriptive that you'll recognize later. 
- Under **Targets**, select **Specified target tags** and enter the tag you created in Step 1 (`whale-speech-inference`).
- Under **Source filter**, select **IPv4 ranges** and enter your desired host IP (the one the app uses, default is `0.0.0.0`).
- Under **Protocols and ports**, select **Specified protocols and ports**, **TCP**, and enter a range of ports you want to open. Here, I used `5000-5005`, but really only 1 is needed. 
- Click **Create**.

It make take a few minutes to update, but the firewall rule should be live shortly.
Since we haven't yet done anything on our instance, I'd reccomend just restarting it to make sure the firewall rule is applied. 

### 3. Pull and run the Docker image
- SSH into your instance. This can be done two different ways:
    - Directly in the GCP console by clicking the SSH button on your instance. A window should pop up with a terminal.
    - Via the command line with `gcloud compute ssh your_gcp_username@your_instance_name --zone your_zone --project your_project`.
- Install Docker on your instance. This can be done with the below commands. Alternatively, you can follow the steps linked above to install Docker on your local machine.
```bash
sudo apt-get update
sudo apt-get install docker.io
```
- Pull the Docker image from Docker Hub. I needed root access when running Docker command, using the `sudo` command. If you skipped Step 0, you can use the pre-built image here, as shown below:
```bash
sudo docker pull permortenhalvorsen024/whale-speech:model-server-x.y.z
```
- Run the Docker image. This will start the model server on the specified port. 
```bash
sudo docker run -p 5000:5000 --name model-server --platform linux/amd64 -it permortenhalvorsen024/whale-speech:model-server-v0.0.2
```

With the container created, and exposed, you should be able to log out of your ssh tunnel without disrupting the model server.


### 4. Test model server

If you are using the default image (where logging is enabled), you should see that your server starts up and is listening on the specified port.
The IP address stated there is only the _internal_ IP address. You'll need to find the _external_ IP address of your instance to access the server from outside the instance.
This can be found in the GCP console, or directly in your ssh termial with the command:
```bash
curl ifconfig.me
```

Using the returned external IP address, you can test that your model running in your Google Compute Engine is accessible to the outside world by running the below command on your local machine:
```bash
python examples/test_model_server.py --model-uri your_external_ip --port 5000
```

You should get a result similar to the below:
```bash
{
    'key': 'encouter1', 
    'predictions': [[0.006112830247730017], ...]
}
```

If you get an error, I suggest going through each step again, to see if you made any changes to the workflow.
If you still are experiencing issues, feel free to open an issue in this repo. 

## Conclusion
Your server should now be up and running, and ready to be used in the pipeline.
You can use this external IP address as the model uri when running the pipeline directly. 

Note, that what we set up here will incrue costs on your GCP account.
So, don't forget to shut down your instance when you are no longer using it.

## Next steps 
Some next steps on where to take your project from here:
- Set up a domain name for your model server.
- Set up HTTPS 
- Wrap these steps into a terraform script for easy reproducibility and portability. For more on IaC, check out the [GCP docs](https://cloud.google.com/deployment-manager/docs/quickstart).


