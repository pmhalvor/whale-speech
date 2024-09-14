# Pipeline order
The question to be answered here is whether detections or sightings search should be run first. 

## Background
Initially, it made sense to only listen for audio detections when whales were sighted in an area. 
This would save costs, by avoiding processing audio data that was not likely to contain whale vocalizations.
However, this approach assumes that whales make vocalizations when they breach the surface, which is the only time a sighting can be made.
This is not always the case, as whales can be silent when they breach the surface, or vocalize when they are not visible.

Since I'm still learning this field, I need to make a decision on which stage to run first. 

## Options

### Detections first

#### Pros
- Saves costs by only processing audio data when a whale is detected.
- Can be used to detect whales that are not visible.
- More likely to find high scoring detections/ model scores (fine-tooth combing)

#### Cons
- Downloading the audio for the 24/7 stream is a heavy task. 
- Would need to crawl all audio data to find detections.
- Not all detections will have images.

### Sightings first

#### Pros
- Can be used to verify detections.
- Images of whales can be attached to found detections.
- Potenitally find new vocalization frequencies not currently covered by detection filters. 

#### Cons
- Not all whale sightings will have corresponding audio.
- More likely to find low scoring detections/ model scores
- Distance from sighting can be too far from hydrophone, even when constraints are set.
    - TODO: Check if there are any restrictions on boating around the Monterey Bay Hydrophone.