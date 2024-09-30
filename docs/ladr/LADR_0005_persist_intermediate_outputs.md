# Persisting intermediate stage outputs

This doc discusses storing outputs from each stage or keeping stateless until the end, and where to do each in our pipeline. 

Some stages (like geo-search) require storing outputs, since the (unaltered) Happywhale API currently writes found encounters to file, and does not return a df.  
<!-- TODO open an PR in happywhale to return df instead of saving to file -->
Other stages like audio retrival may make sense to keep stateless to avoid storage costs on our end. 
For debugging purposes, storing these data makes sense, since it speeds up iteration time. 
But a productionized pipeline might only run once per geofile-date, and not need to store intermediate outputs.

Another point to consider is that data-usage agreements with the audio proivders.
The audio from MBARI is ok to download, but what happens when we scale to include other hydrophone sources? 


Some questions to consider:
- Do we want to store outputs from any intermediate stage, or just the last one? Ex. start/stop times of sifted audio, full classification arrays or pooled results.

- How to handle overwrites or parallel writes? Parallel writes should never occur, since we find overlapping encounter ids, and group them together. Overwrites could occur if stage does not check if data exists for stage before writing.
<!-- TODO add unit test for parallel writes -->
- Do we have a true key throughout our entire dataflow? Do we need one? After geo-search, we could consider a concatenation of start, end, and encounter_id as key, though this might be misleading, if sifted audio changes the start and end times. 




Interesting blog post discussing some of these issues, comparing stateful DoFns to CombineFn: https://beam.apache.org/blog/stateful-processing/

## Stages
For every stage, I'll discussion the necessity of storing outputs, and if so, what outputs to store.

### 1. Geo-search
We are forced to store the outputs for this stage. 
Maybe outputs should be written to a database to avoid duplicates, but is this overengineering? 

What was the problem with proivding a ByteIO object as the export file? That could be used to convert the data to a df, and then distribute how we see fit, instead of loading the data from a file and passing onward into the pipeline.


### 2. Audio retrieval
We should likely not store the outputs for this stage.
The data is open source and can be retrieved at any time, only costs to download.
The main argument for storing here would be if download costs were significantly higher than storage.
For now, we assume that this pipeline will be run on new servers often, mostly once per date, meaning storing audio does is not worth it.


### 3. Audio sifting
How much audio sift data should be persisted? Full audio arrays with start, stop times and encounter ids? 
Or just the start, stop times and encounter ids, assuming the audio will be downloaded and passed from the previous stage?

The main argument for storing the full audio arrays is that it speeds up iteration time, and allows for easier debugging.
We likely also want this data easily accessible, if our final outputs are going to contain the audio snippets with classifications. 
That's kinda the whole point of this pipeline, so _some_ audio will eventually need to be be stored. 
And its likely at this stage we will want to store it, since this audio is truncated.


### 4. Classification
After the audio has been fed through the model, we'll get an array shorter than the length of the audio array, but linearally scaled to the audio. So larger context windows will eventually start producing very large classification arrays. Are all of these data necessary to save, or would a pooled score be best? It depends on the use-case ;) 

We could alternatively cut the audio to only the parts where a min and max classification above a threshold is found. 
This would eliminate any real dependency on audio sifting (in case that stage turns out to not be needed later). 
And This would serve as the best waste-reduction strategy, since we would only store the audio that we are confident contains a whale call.


### 5. Postprocessings (pooling and labelling)
The final stage definitely needs to be store, but the main discussion here becomes, what to store? 
If we already have stored intermediate stages like sifted audio or truncated classified audio, we could avoid saving them again, and rather load from those tables when presenting aggregated results. 

Though, I like the idea of the last stage of the pipeline containing all the data necessary found through the pipeline. This makes data sharing easier, with a concrete final product, instead of a bunch of fragmentated tables that need to be joined to have any true value. 