import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, DirectOptions, GoogleCloudOptions, StandardOptions
from src.pipeline.stages.search import GeometrySearch
from src.pipeline.stages.audio import RetrieveAudio, WriteAudio, WriteSiftedAudio
from src.pipeline.stages.sift import Butterworth
from src.pipeline.stages.classify import HumpbackWhaleClassifier

from src.pipeline.config import load_pipeline_config
config = load_pipeline_config()

from apache_beam.runners.interactive.cache_manager import FileBasedCacheManager

FileBasedCacheManager().cleanup()

def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(SetupOptions).save_main_session = True
    # options.view_as(DirectOptions).direct_num_workers = 1  # Limit to 1 worker
    
    if not config.general.local:
        # Set up Google Cloud options
        google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
        google_cloud_options.project = 'bioacoustics-2024'
        google_cloud_options.job_name = 'whale-speech-sift-classify-local'
        google_cloud_options.staging_location = 'gs://bioacoustics/whale-speech/staging'
        google_cloud_options.temp_location = 'gs://bioacoustics/whale-speech/temp'
        google_cloud_options.region = 'us-central1'  # Select your region

        # Use the Dataflow runner instead of the local DirectRunner
        pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'

    # set requirements file
    pipeline_options.view_as(SetupOptions).requirements_file = 'requirements.txt'
    pipeline_options.view_as(SetupOptions).setup_file = './setup.py'
    pipeline_options.view_as(SetupOptions).extra_packages = ['whale-speech-0.0.1.zip']
    
    args = {
        "start": config.input.start,
        "end": config.input.end
    }

    with beam.Pipeline(options=pipeline_options) as p:
        input_data        = p               | "Create Input"        >> beam.Create([args])  
        search_results    = input_data      | "Geometry Search"     >> beam.ParDo(GeometrySearch(config))
        
        audio_results     = search_results  | "Retrieve Audio"      >> beam.ParDo(RetrieveAudio(config))
        # audio_files       = audio_results   | "Store Audio (temp)"  >> beam.ParDo(WriteAudio())

        sifted_audio      = audio_results   | "Sift Audio"          >> Butterworth(config)
        # sifted_audio_files = sifted_audio   | "Store Sifted Audio"  >> beam.ParDo(WriteSiftedAudio("butterworth"))

        classified_audio  = sifted_audio    | "Classify Audio"      >> HumpbackWhaleClassifier(config)
        
        
        # For debugging, you can write the output to a text file
        # audio_files     | "Write Audio Output"  >> beam.io.WriteToText('audio_files.txt')
        # search_results  | "Write Search Output" >> beam.io.WriteToText('search_results.txt')
        # sifted_audio_files | "Write Sifted Audio Output" >> beam.io.WriteToText('sifted_audio_files.txt')

        # # Post-process the labels
        # postprocessed_labels = classified_audio | "Postprocess Labels" >> PostprocessLabels()

        # Output results
        # postprocessed_labels | "Write Results" >> beam.io.WriteToText("output.txt")

if __name__ == "__main__":
    import cProfile

    cProfile.run('run()')
    # run()
