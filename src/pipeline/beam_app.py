import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from beam_stage_search import GeometrySearch
# from beam_stage_audio import RetrieveAudio


def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(options=pipeline_options) as p:
        input_data =        p               | "Create Input" >> beam.Create([{'start': '2016-12-21T00:30:0', 'end':"2016-12-21T00:40:0"}])  
        search_results =    input_data      | "Run Geometry Search" >> beam.ParDo(GeometrySearch())
        # audio_results  =    search_results  | "Retrieve Audio" >> RetrieveAudio()




        # For debugging, you can write the output to a text file
        search_results | "Write Output" >> beam.io.WriteToText('output.txt')


        # Perform Geometry Search and Retrieve Audio
        # geometry_results = p | "Geometry Search" >> GeometrySearch(pipeline_options)
        # audio = geometry_results | "Retrieve Audio" >> RetrieveAudio()

        # # Filter Frequency based on the audio and classify it
        # filtered_audio = audio | "Filter Frequency" >> FilterFrequency()
        # classified_audio = filtered_audio | "Classify Audio" >> ClassifyAudio()

        # # Post-process the labels
        # postprocessed_labels = classified_audio | "Postprocess Labels" >> PostprocessLabels()

        # Output results
        # postprocessed_labels | "Write Results" >> beam.io.WriteToText("output.txt")

if __name__ == "__main__":
    run()
