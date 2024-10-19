# Model Server Deployment

Need to decide how to host the model server.

When running locally, I've been starting a Flask server in a separate thread. 
For my production environment, I will likely need a more robust solution. 

The desired solution should:
- Be fast, scalable, and stable
- Be cost-effective
- Be easy to deploy and maintain
- Have some version control

## Options

### 1. Compute Engine VM
**Pros**: 
- Full control over the environment
    - Easy to debug by ssh-ing into the VM
    - Manually install and update needed dependencies
- Very similar to local development
- Can host multiple services on the same VM
    - Ex. if inference server and pipeline triggers were on the same VM

**Cons**:
- Requires more setup and maintenance
    - Networking Firewall rules in GCP
    - Monitoring and logging not built-in
- Not as scalable as other options
- Persistent servers would likely be more expensive than serverless options

### 2. Cloud Run
**Pros**:
- Serverless
    - Only pay for what you use
    - Scales automatically
- Easy to deploy
    - Can deploy and revise directly from `gcloud` or in the GCP console
- Built-in monitoring and logging
- Built-in version control (using image registry and/or tags)
- Exposes a public endpoint that can be triggered by HTTP requests

**Cons**:
- Can only serve one contianer per service. Other services would need to be deployed separately.
- Haven't figured out how to scale up (to recieve large input requests)

### 3. Kubernetes Engine
**Pros**:
- Full control over the environment
- Scalable
- Can host multiple services on the same cluster

**Cons**:
- Takes a (relatively) long time to start and scale up 
- Requires more setup and maintenance
- Not as cost-effective as serverless options
- Probably overkill for this project


## Decision
For this project, I'll use Cloud Run.
I tried a VM first, but realized it costs too much over time, and missed the ability to easily scale.

Cloud Run worked pretty much out of the box, and I was able to deploy the model server in a few minutes.
Figuring out the correct PORT configuration was a bit cumbersome, though. 

I think the stateless nature will be a cheapest option for the end goal of this project. 
During times of high activity, we can keep the minimum instance count at 1, to ensure faster response times.
Otherwise, we can scale down to 0 instances, and only pay for the storage of the container image (if using Artifact Registry).

I just need to figure out how to scale up the instances to handle larger requests.
