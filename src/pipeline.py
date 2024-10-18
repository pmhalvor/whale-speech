import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from stages.search import GeometrySearch
from stages.audio import RetrieveAudio
from stages.sift import Butterworth
from stages.classify import WhaleClassifier
from stages.postprocess import PostprocessLabels


from config import load_pipeline_config
config = load_pipeline_config()

def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions(
        runner="DataflowRunner",
        region="us-central1",
        project=config.general.project,
        temp_location=config.general.temp_location,
    )
    pipeline_options.view_as(SetupOptions).save_main_session = True
    args = {
        "start": config.input.start,
        "end": config.input.end
    }

    with beam.Pipeline(options=pipeline_options) as p:
        input_data      = p                 | "Create Input"        >> beam.Create([args])  
        search_output   = input_data        | "Run Geometry Search" >> beam.ParDo(GeometrySearch(config))
        audio_output    = search_output     | "Retrieve Audio"      >> beam.ParDo(RetrieveAudio(config))
        sifted_audio    = audio_output      | "Sift Audio"          >> Butterworth(config)
        classifications = sifted_audio      | "Classify Audio"      >> WhaleClassifier(config)
        pipeline_output = classifications   | "Postprocess Labels"  >> beam.ParDo(
            PostprocessLabels(config),
            search_output=beam.pvalue.AsSingleton(search_output),
            audio_output=beam.pvalue.AsList(audio_output),
            sifted_audio=beam.pvalue.AsList(sifted_audio),
        )


if __name__ == "__main__":
    run()
