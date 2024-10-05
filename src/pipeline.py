import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from stages.search import GeometrySearch
from stages.audio import RetrieveAudio, WriteAudio, WriteSiftedAudio
from stages.sift import Butterworth
from stages.classify import WhaleClassifier, WriteClassifications
from stages.postprocess import PostprocessLabels, WritePostprocess

from apache_beam.io.gcp.internal.clients import bigquery


from config import load_pipeline_config
config = load_pipeline_config()

def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions(
        # runner="DataflowRunner",
        project=config.general.project,
        temp_location=config.general.temp_location,
    )
    pipeline_options.view_as(SetupOptions).save_main_session = True
    args = {
        "start": config.input.start,
        "end": config.input.end
    }

    with beam.Pipeline(options=pipeline_options) as p:
        input_data          = p                 | "Create Input"        >> beam.Create([args])  
        search_output       = input_data        | "Run Geometry Search" >> beam.ParDo(GeometrySearch(config))
        audio_output        = search_output     | "Retrieve Audio"      >> beam.ParDo(RetrieveAudio())
        sifted_audio        = audio_output      | "Sift Audio"          >> Butterworth()
        classifications     = sifted_audio      | "Classify Audio"      >> WhaleClassifier(config)
        postprocess_labels  = classifications   | "Postprocess Labels"  >> beam.ParDo(
            PostprocessLabels(config),
            search_output=beam.pvalue.AsSingleton(search_output),
        )

        # Store results
        audio_output        | "Store Audio (temp)"      >> beam.ParDo(WriteAudio())
        sifted_audio        | "Store Sifted Audio"      >> beam.ParDo(WriteSiftedAudio("butterworth"))
        classifications     | "Store Classifications"   >> beam.ParDo(WriteClassifications(config))
        postprocess_labels  | "Write to BigQuery"       >> beam.ParDo(WritePostprocess(config))


if __name__ == "__main__":
    run()
