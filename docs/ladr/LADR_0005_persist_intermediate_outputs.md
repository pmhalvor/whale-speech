# Intermediate stage outputs

This doc discusses handling the intermediate outputs between stages. 
Whether or not to store the output should be confirgurable for the end user, either via command line params or in [config.yaml](../../src/config/common.yaml)
We want to enable these storage options to support local adn cloud storage. 
This means we need to consider costs and effeciency when designing output schemas. 

Some stages (like geo-search) require locally storing outputs, since the (unaltered) Happywhale API currently writes found encounters to file, and does not return a df.  <!-- TODO open an PR in happywhale to return df instead of saving to file -->

Other stages like audio retrival may make sense to keep stateless to avoid storage costs on our end. 
Or storing only start and stop values for the audio, with a link to that day's data. 


For during local development and debugging, having all intermediate data stored helps speed up iteration time.
Additionally, if data already exists for run on a particular date, the pipeline should skip these stages and load the outputs from the previous run. 
While a productionized pipeline might only run once per geofile-date, and not need to store intermediate outputs, this decision should be left up to the end user.

Another point to consider is that data-usage agreements with the audio proivders.
Make sure to read any agreements to ensure that storing intermediate outputs is allowed.
<!-- TODO build document for keeping track of hydrophone data usage agreements. -->


Some questions to consider:
- Exactly what data should we preserve from each stage? Will this different from the output of the stage?
Ex. start/stop times of sifted audio, full classification arrays or pooled results.

- How to handle overwrites or parallel writes? Parallel writes should never occur, since we find overlapping encounter ids, and group them together. Overwrites could occur if stage does not check if data exists for stage before writing.
<!-- TODO add unit test for parallel writes -->

- Do we have a true key throughout our entire dataflow? Do we need one? After geo-search, we could consider a concatenation of start, end, and encounter_id as key, though this might be misleading, if sifted audio changes the start and end times. 

## Stages
For every stage, I'll discuss what outputs to store, and how they should be written locally and in the cloud.

### 1. Geo-search
#### Local storage
We are forced to store the outputs for this stage, since the API requires a file path to write the pandas df to. 
This means, there is a chance of overwites when running locally, or on a persistant server.
Can however be solved by providing a temporary file location to the API, loading in the old data and the temporary outputs, then write to the final location.  

#### Cloud storage
This data is very structured, and could be stored in a database.
We should init and create a table in our project.dataset_id. 
This can be the alternative to storing the temporary file to a more persistant final location.


### 2. Audio retrieval
We should likely not store the full outputs for this stage.
The data is open source and can be retrieved at any time, only costs to download.
The main argument for storing here would be if download costs were significantly higher than storage, i.e. on a persistant server.
Development and debugging are still easier with the data stored, so we also need to smartly design these outputs. 

#### Local storage
Writing to my local machine has been easy enough with np.save. 
This makes the data easily accessible for listening to, which is extremely helpful when analysing plots. 
For now, I'll assume this is good enough, and rather rely on the built-in teardown of the DataflowRunner to clean this data up if wronglly configured during cloud runs. 

#### Cloud storage
We could store the start, stop times of (and maybe url link to) the audio retrived for the found encounter ids in a table in our project.dataset_id.
This will can be beneficial if a user decides not to use any audio sifting. 
Maybe add config option to allow storing full audio? 
If stored, should be identified by a key (start, stop, encounter_id) to avoid overwrites, and stored in a bucket, not a table.

### 3. Audio sifting
How much audio sift data should be persisted? 
Full audio arrays with start, stop times and encounter ids? 
Or just the start, stop times and encounter ids, assuming the audio will be downloaded and passed from the previous stage?
There is really no need to double storage here, but option should still be available. 

The main argument for storing the full audio arrays is that it speeds up iteration time, and allows for easier debugging.
We likely also want this data easily accessible, if our final outputs are going to contain the audio snippets with classifications. 
That's kinda the whole point of this pipeline, so _some_ audio will eventually need to be be stored. 
And its likely at this stage we will want to store it, since this audio is truncated.

Will need to think about the key or unique identifier here, since there are a lot of parameters that can affect how much audio was truncated. essentially, all of these can be in the path, but that will make for extremely long paths. 

#### Local storage
Again, np.save is a good option for storing the audio arrays.

#### Cloud storage
Similar as before, needs to be stored in a bucket. 
Can maybe inherit same write method from previous stage, if we fingure out how to pass classes between stage local stages-files. 

### 4. Classification
After the audio has been fed through the model, we'll get an array shorter than the length of the audio array, but still arbitrry lengths. 
Large context windows will produce large classification arrays, meaning high storage costs. 
Are all of these data necessary to save, or would a pooled score be best? It depends on the use-case ;) 

We could alternatively cut the audio to only the parts where a min and max classification above a threshold is found. 
This would eliminate any real dependency on audio sifting (in case that stage turns out to not be needed later). 
And This would serve as the best waste-reduction strategy, since we would only store the audio that we are confident contains a whale call.

#### Local storage
For now, let' stick to storing the entire clasasification array for this stage, using np.save, with similar paths to audio storage.

#### Cloud storage
Since we are dealing with arbitrary lengths, I'd say stick to bucket with parameters as path variables. 


### 5. Postprocessings (pooling and labelling)
The final stage definitely needs to be store, but the main discussion here becomes, what to store? 
If we already have stored intermediate stages like sifted audio or truncated classified audio, we could avoid saving them again, and rather load from those tables when presenting aggregated results. 

Though, I like the idea of the last stage of the pipeline containing all the data necessary found through the pipeline. 
This makes data sharing easier, with a concrete final product, instead of a bunch of fragmentated tables that need to be joined to have any true value. 
Maybe storing the easily queryable data in a table, then include a link to the audio storage location (whether that be a file by me or MBARI or other hydrophone provider).

#### Local storage
I'll assume a similar structure to the expected tobale when saving local. This means arrays data like audio and classifications are excluded. 
This frees me up to store one entry per encounter id. 
Paths to the audio and classification arrays can be stored in the table.

#### Cloud storage
I feel like this table shuold always be written to (i.e. no config option to disable).
Outputs will include:
```
encounter_id
longitude
latitude
start_time
stop_time
pooled_score
img_path
audio_path 
classification_path
```

## Conclusion
- All stages (except last) will be configurable to save or not. 
- File exists sensors should allow skipping a stage.
- Local storage or arrays w/ variable lengths with np.save, and paramter values in path.
- Structured data will be stored in tables with relevant links to array data


## Resources
- Interesting blog post comparing stateful DoFns to CombineFn: https://beam.apache.org/blog/stateful-processing/
- Beam BigQuery walkthrough: https://beam.apache.org/documentation/io/built-in/google-bigquery/
