# Lightweight Detection Mechanisms

This doc's purpose is to consider the different options for simple whale vocalization detection mechanisms.

## Background
The purpose of this pipeline to is efficiently detect vocalizations from whales encountered on [HappyWhale](https://happywhale.com/). 
Since audio data is notoriously large, we want to quickly find which chunks of audio are most important to look at, before classifying them via the [humpback_whale model](https://tfhub.dev/google/humpback_whale/1).

Initially, the data that does not make it past the filter will not be fed through the machine learning model, in order to keep costs down. 
This means, we need a filtering mechanism that is "generuous" in what it flags, but not too generous that it flags everything.

## Options

### Energy filter
Simplest of filters. Just measures peaks in audio signal, above a specified threshold.

#### Pros
- very lightweight
- easy to implement

#### Cons
- too much noise can make it through
- not very specific
- prioritizes loudness over frequency
- sounds from a distance will likely not be detected


### Butterworth Bandpass Filter
[Wikipedia](https://en.wikipedia.org/wiki/Butterworth_filter)

Filters out audio that does not contain a certain frequency range.
Only allows a particular band of frequencies to pass through. 
These are determined via the filter's order and cutoff frequencies, low and high.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Bandwidth.svg/320px-Bandwidth.svg.png)

#### Pros 
- more specific than energy filter
- can be tuned to specific frequencies, i.e. a specific species
- lightweight
- fast (~6 s to computer for 30 min)
- easy to implement
- can be used together w/ other filtering methods

#### Cons
- room for improvement on specificity
- assumes a certain frequency range is the most important 
    - disregards harmonics
    - clicks may not be detected
    - different ages/individuals may have different frequency ranges (?)
- not great at detecting sounds from a distance


### Spectrogram
A visual representation of the spectrum of frequencies of a signal as it varies with time.
This is a 2D representation of the audio signal, where the x-axis is time, y-axis is frequency, and color is amplitude.

#### Pros
- can be used to detect harmonics
- can be used to detect clicks
- can be used to detect sounds from a distance
- can be used to detect multiple species

#### Cons
- computationally expensive
- not lightweight
- more difficult to work with (2D data)


### Humpback Whale Model
Final option is to just directly use the model on the data surronding a encounter. 
This is the most expensive option, but also the most accurate.

#### Pros
- most accurate
- "smartest" filter

#### Cons
- computationally expensive
- slowest


## Decision
No true decision has been made yet. 
I want to run a few more experiments on encounters different distances from the hydrophone, and see how the different filters perform.
The results fromt hat experiment can be documented here, and will help steer the decision.

 
