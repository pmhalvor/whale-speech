import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from stages.search import GeometrySearch
from stages.audio import RetrieveAudio, WriteAudio, WriteSiftedAudio
from stages.sift import Butterworth
from stages.classify import WhaleClassifier, WriteClassifications
from stages.postprocess import PostprocessLabels


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
        input_data          = p                 | "Create Input"        >> beam.Create([args])  
        search_output       = input_data        | "Run Geometry Search" >> beam.ParDo(GeometrySearch())
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
        postprocess_labels  | "Write Results"           >> beam.io.WriteToText("data/output.txt", shard_name_template="")

        # Output results
        # postprocessed_labels | "Write Results" >> beam.io.WriteToText("output.txt")


if __name__ == "__main__":
    run()
