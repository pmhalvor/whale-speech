import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from stages.search import GeometrySearch
from stages.audio import RetrieveAudio, WriteAudio, WriteSiftedAudio
from stages.sift import Butterworth
from stages.classify import WhaleClassifier

from config import load_pipeline_config
config = load_pipeline_config()

def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(SetupOptions).save_main_session = True
    args = {
        "start": config.input.start,
        "end": config.input.end
    }

    with beam.Pipeline(options=pipeline_options) as p:
        input_data      = p             | "Create Input"        >> beam.Create([args])  
        search_output   = input_data    | "Run Geometry Search" >> beam.ParDo(GeometrySearch())
        
        audio_output    = search_output | "Retrieve Audio"      >> beam.ParDo(RetrieveAudio())
        audio_files     = audio_output  | "Store Audio (temp)"  >> beam.ParDo(WriteAudio())

        sifted_audio    = audio_output  | "Sift Audio"          >> Butterworth()
        sifted_audio_files = sifted_audio   | "Store Sifted Audio"  >> beam.ParDo(WriteSiftedAudio("butterworth"))

        classifications = sifted_audio  | "Classify Audio"      >> WhaleClassifier(config)


        # # Post-process the labels
        # postprocessed_labels = classified_audio | "Postprocess Labels" >> PostprocessLabels()

        # Output results
        # postprocessed_labels | "Write Results" >> beam.io.WriteToText("output.txt")

        # For debugging, you can write the output to a text file
        # audio_files     | "Write Audio Output"  >> beam.io.WriteToText('audio_files.txt')
        # search_results  | "Write Search Output" >> beam.io.WriteToText('search_results.txt')


if __name__ == "__main__":
    run()
