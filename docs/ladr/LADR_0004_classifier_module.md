# Desiging the classifier module

During local development, we ran into issues using our pretrained TensorFlow model inside a `beam.DoFn`.
Running the model in an isolated script worked fine, with the capability to handle large inputs,
but for some reason, running through Beam was problmeatic. 
Research tells me this is either due to a memory allocation issue or model serialization issue. 

Either way, a work around is needed to enable local development (for debugging purposes) that's closely coupled to our expected cloud-based production environment.

## Options

### Option 1: Use a smaller model
I found a quantized model the seemingly condenses the [google/humpback_whale model](https://tfhub.dev/google/humpback_whale/1)  size enough to run in Beam, made by Oleg A. Golev (oleggolev) at https://github.com/oleggolev/COS598D-Whale/. 
The original model is converted to a tflite model with slightly adapted input and output layers. 
Example code for handling this model can be found at [examples/quantized_model.py](../../examples/quantized_model.py) and [examples/quantized_inference.py](../../examples/quantized_inference.py).

#### Pros
- actually works in Beam (on my local machine)
- could speed up inference time and potentially reduce overall costs
- originally quantized to be deployed on small edge devices, should be portable to most environments
- model files easily downloadable (present in GitHub repo)
- keeps all our processing in one single unit -> cleaner project structure on our end

#### Cons
- initial findings found classifications on most random arrays of dummy data -> too many false positives (I could be wrong here. Track issue: https://github.com/oleggolev/COS598D-Whale/issues/1)
- committing to this set-up restricts us to a fixed model size
- not easily swapped out for new models or architectures -> requires quantization of each new model used (high maintaince)
- expected input size correlates to 1.5 seconds of audio, which feels too short to correctly classify a whale call (I may be mistaken here though)
- outputs have to be aggregated for every 1.5 seconds of audio -> more post-process compute than original model
- poorly documented repository, doesn't feel easy to trust right off the bat


### Option 2: Model as a service
Host the model on an external resource, and call it via an API.

#### Pros
- model easily be swapped out, updated, monitored, and maintained
- with an autoscaler, the model server can handle larger inputs or even multiple requests at once
- endpoint can be easily accesible to other developers (if desired)
- error handling and retries won't initially break the processing pipeline (ex. 4 retries w/ exponential backoff then return no classifications found)
- build personal exprience with exposing models as services
- external compute allows the ML framework (TF, ONNX, Torch, etc) to manage memory how it wants to, instead of constraints enforced by Beam
- reduces pipeline dependencies (though project dependencies remain same)

#### Cons
- fragments the codebase -> pipeline not easily packaged as a single unit which makes portability and deployment more difficult
- requires to be running on two resources instead of one
- likely more expensive (though some research around model hosting/serving options may find a cost-effective solution)
- requires integration with more cloud services (doubled-edged sword, since this also gives me more experience with other cloud tools)

### Option 3: Continue w/o ability for local development
Since the model is intended to run in the cloud anyway, we can use this motivation to push toward cloud-only development. 

#### Pros
- can continue development as already written, following same structure as rest of pipeline
- keeps all processing in one single unit

#### Cons
- debugging is more difficult
- lack of local testing makes development more time-consuming (waiting for deploys etc)
- feels very "brute-force" to just throw more resources at the problem instead of reevaluating
- restricts development to high-resource environments -> expensive development

## Decision
I'm going to go with Option 2: Model as a service.
This is by far the best choice, though I wanted to give a far chance to exploring other options. 
More ideas can be added underway, but option 2 is the most flexible and scalable option.
Any additional costs can be mitigated by optimizing the model server or implementing an efficient teardown strategy.
