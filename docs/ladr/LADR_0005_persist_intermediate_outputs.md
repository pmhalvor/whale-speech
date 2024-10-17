# Intermediate stage outputs

This doc discusses handling the intermediate outputs between stages. 
Whether or not to store the output should be confirgurable for the end user, either via command line params or in [config.yaml](../../src/config/common.yaml).
We want to support both local and cloud storage. 
This means we need to consider costs and effeciency when designing output schemas. 

Some stages (like geo-search) require locally storing outputs, since the (unaltered) Happywhale API currently writes found encounters to file, and does not return a df.  <!-- TODO open an PR in happywhale to return df instead of saving to file -->

Other stages like audio retrival may make sense to keep stateless to avoid storage costs on our end. 
Or storing only start and stop values for the audio, with a link to that day's data. 


During local development and debugging, having all intermediate data stored helps speed up iteration time.
If data already exists for run on a particular date w/ same config params, the pipeline should skip these stages and load the outputs from the previous run. 
A productionized pipeline may likely only run once per geo-file date, depending on use-case, but storage decisions should be left up to the end user.

Another point to consider is data-usage agreements with the audio proivders.
Make sure to read any agreements to ensure that storing intermediate outputs is allowed.
<!-- TODO build document for keeping track of hydrophone data usage agreements. -->


Some questions to consider:
- Exactly what data should we persist from each stage? Will these data differ from the output of the stage?
Ex. start/stop times of sifted audio, full classification arrays or pooled results.

- How to handle overwrites or parallel writes? Parallel writes should never occur, since we find overlapping encounter ids, and group them together, and treat as single row in downstream stages. Overwrites could occur if stage does not check if data exists for stage before writing. For the most part, loading previous and appending new data is likely the best strategy here.
<!-- TODO add integration test for parallel writes -->

- Do we have a true unique identifiable key throughout our entire dataflow? Do we need one? After geo-search, we could consider a concatenation of start, end, and encounter_id as key, though this might be misleading, if sifted audio changes the start and end times. 

## Stages
For every stage, I'll discuss what outputs to store, and how they should be written locally and in the cloud.

### 1. Geo-search

Columns to persist:
```
encounter_id (str)
encounter_time (iso formatted datetime)
longitude (float)
latitude (float)
displayImgUrl (str)
```
These works well for both local and cloud storage options. 


#### Local storage

We are forced to store the outputs for this stage, since the API requires a file path to write the pandas df to. 
This means, there is a chance of overwites when running locally, or on a persistant server.
Can however be solved by providing a temporary file location to the API, loading in the old data and the temporary outputs, then write to the final location.  


#### Cloud storage
This data is very structured, and could be stored in a database.
We should init and create a table in our project.dataset_id. 
This will be the alternative to storing the temporary file to a more persistant final location when `is_local=False`. 


### 2. Audio retrieval
We should likely not store the full outputs for this stage.
The data is open source and can be retrieved at any time, only costs to download.

An argument for storing here would be if download costs were significantly higher than storage costs, for example on a persistant server.
Development and debugging are still easier with the data stored, so we also need to smartly design these outputs. 

Another arguement would be to have the original signal that was fed into the downstream stasks. 

Columns to persist in table:
```
key (str) 
audio_path (str)
start_time (str, iso formatted datetime) NULLABLE
stop_time (str, iso formatted datetime) NULLABLE
```

To persist in bucket as array (key in path):
```
audio (np.array) NULLABLE
```

#### Local storage
Writing array to local machine is easy enough with np.save, with date and key in path. 
This makes the data easily accessible for listening to, which is extremely helpful when analysing plots. 

Additionally, we can maintain an index-table for the audio stored locally, with the columns mentioned above.

#### Cloud storage
Store the start, stop times of (and maybe url link to) the audio retrived for the found encounter ids in a table in our project.dataset_id.
This will can be beneficial if a user decides not to use any audio sifting. 

Uploading audio to its own bucket can be configurable (with default False).
If stored, should be identified by a key (start, stop, encounter_id) to avoid overwrites, and stored in a bucket, not a table.

### 3. Audio sifting
How much sifted audio data should be persisted? 
Full audio arrays with start, stop times and encounter ids? 
Or just the start, stop times and encounter ids, assuming the audio will be downloaded and passed from the previous stage?
There is really no need to double storage here, but option should still be available. 

We likely want this data easily accessible, since it is the input to our inference model. 
Our final pipeline outputs will likely want to display these data as audio snippets with classifications and images. 
That's kinda the whole point of this pipeline, so _some_ audio will eventually need to be be stored. 
And its likely at this stage we will want to store it, since this audio is truncated.

Will need to think about the key or unique identifier here, since there are a lot of parameters that can affect how much audio was truncated. 
Essentially, all of these can be in the path, but that will make for extremely long paths. 
However, the long path makes it very explicit how the audio was parsed (ex. `.../filter=butterworth/highcut=1500/lowcut=50/order=5/...`)
Most params are not hierarchical, though using a wildcard `*` in path on read helps avoid this issue (ex. `.../filter=butterworth/highcut=*/lowcut=50/order=5/...`). 

Columns to persist in table:
```
key (str)
audio_path (str)
detections_path (str)
params (Dict[str, str], might need to be str in queryable-table)
```

To persist in bucket as inidividual arrays (params and key in path):
```
audio (np.array)
detections (np.array)
```


#### Local storage
Again, np.save is a good option for storing the audio arrays.
An index-table can be maintained in parent dir in same path, or in own query table.

#### Cloud storage
Similar as before, needs to be stored in a bucket w/ params in path. 

### 4. Classification
After the audio has been fed through the model, we'll get an array shorter than the length of the audio array, but still arbitrary lengths. 
Large context windows will produce large classification arrays, meaning high storage costs. 
Whether or not all of these data necessary to save, or if a pooled score would be best, depends on the use-case. 😉

We could alternatively cut the audio to only the parts where a min and max classification above a threshold is found.
This requires even more processing, butwould eliminate any real dependency on audio sifting (in case that stage turns out to not be needed later). 
Such a waste-reduction strategy is the most efficient means of writing this, since we would only store the audio that we are confident contains a whale call.
I'm just hesitant that we risk cutting too much audio, especially in cases where the model is unsure at the beginning or end of a signal.

Columns to persist in table:
```
key (str)
classification_path (str)
```

To persist in bucket as array (params in path, similar to audio):
```
classification (np.array)
```


#### Local storage
For now, let' stick to storing the entire clasasification array for this stage, using `np.save`, with similar paths to audio storage.

#### Cloud storage
Since we are dealing with arbitrary lengths, I'd say stick to bucket with parameters as path variables. 
Simliar to audio, maintain a table with the key and path to the classification array.


### 5. Postprocessing (pooling and labelling)
The final stage definitely needs to be stored, but the main discussion here is what to store? 
If we already have stored intermediate stages like sifted audio or truncated classified audio, we could avoid saving them again, and rather load from those tables when presenting aggregated results. 

I like the idea of the last stage of the pipeline containing all the data necessary found through the pipeline. 
This makes data sharing easier, with a concrete final product, instead of a bunch of fragmentated tables that need to be joined to have any true value. 
Maybe storing the easily queryable data in a table, then include a link to the audio storage location (whether that be a bucket-path or local file by me or url to MBARI or other audio sources from hydrophone providers).

Final outputs will include:
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
result_plot_path
```


#### Local storage
I'll assume a similar structure to the expected table when saving local. 
This means arrays data like audio and classifications are excluded, which frees me up to store one entry per encounter id. 
Paths to the audio and classification arrays can be stored in the table.

#### Cloud storage
Mirror the local table in some queryable table in our cloud provider. 

## Conclusion
- All stages will be configurable to save or not, default to true for all (for now). 
- File exists sensors should allow skipping a stage.
- Local storage or arrays w/ variable lengths with np.save, and paramter values in path.
- Structured data will be stored in tables with relevant links to array data


## Resources
- Interesting blog post comparing stateful DoFns to CombineFn: https://beam.apache.org/blog/stateful-processing/
- Beam BigQuery walkthrough: https://beam.apache.org/documentation/io/built-in/google-bigquery/
