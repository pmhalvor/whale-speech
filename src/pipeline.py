import apache_beam as beam

from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from stages.search import GeometrySearch
from stages.audio import RetrieveAudio, WriteAudio, WriteSiftedAudio
from stages.sift import Butterworth
from stages.classify import WhaleClassifier, WriteClassifications
from stages.postprocess import PostprocessLabels

from apache_beam.io.gcp.internal.clients import bigquery


from config import load_pipeline_config
config = load_pipeline_config()

def run():
    # Initialize pipeline options
    pipeline_options = PipelineOptions(
        # runner="DataflowRunner",
        project="bioacoustics-2024",
        temp_location="gs://bioacoustics/whale-speech/temp",
        # region=config.general.region,
        # job_name=config.general.job_name,
        # temp_location=config.general.temp_location,
        # setup_file="./setup.py"
    )
    pipeline_options.view_as(SetupOptions).save_main_session = True
    args = {
        "start": config.input.start,
        "end": config.input.end
    }
    schema = {
        "fields" : [
            # {'name': 'key', 'type': 'STRING', 'mode': 'REQUIRED'}, 
            {'name': 'classifications', 'type': 'FLOAT64', 'mode': 'REPEATED'}, 
            {'name': 'pooled_score', 'type': 'FLOAT64', 'mode': 'REQUIRED'}, 
            {'name': 'encounter_id', 'type': 'STRING', 'mode': 'REQUIRED'}, 
            {'name': 'displayImgUrl', 'type': 'STRING', 'mode': 'REQUIRED'}, 
            {'name': 'longitude', 'type': 'FLOAT64', 'mode': 'REQUIRED'}, 
            {'name': 'latitude', 'type': 'FLOAT64', 'mode': 'REQUIRED'}, 
            {'name': 'start', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}, 
            {'name': 'end', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
        ]
    }
    # table_ref = "{project_id}:{dataset_id}.{table_id}".format(
    #     project_id=config.general.project, 
    #     dataset_id=config.general.dataset_id, 
    #     table_id=config.postprocess.postprocess_table_id
    # )
    # table_spec = bigquery.TableReference(
    #     projectId=config.general.project,
    #     datasetId=config.general.dataset_id,
    #     tableId=config.postprocess.postprocess_table_id
    # )
    table_spec = f"bioacoustics-2024:whale_sppech.mapped_audio"

    print(f"Writing to table: {table_spec}")
    print(f"PipelineOptions: {pipeline_options}")

    with beam.Pipeline(options=pipeline_options) as p:
        input_data          = p                 | "Create Input"        >> beam.Create([args])  
        search_output       = input_data        | "Run Geometry Search" >> beam.ParDo(GeometrySearch())
        audio_output        = search_output     | "Retrieve Audio"      >> beam.ParDo(RetrieveAudio())
        sifted_audio        = audio_output      | "Sift Audio"          >> Butterworth()
        classifications     = sifted_audio      | "Classify Audio"      >> WhaleClassifier(config)
        # postprocess_labels  = classifications   | "Postprocess Labels"  >> PostprocessLabels(config, search_output)
        postprocess_labels  = classifications   | "Postprocess Labels"  >> beam.ParDo(
            PostprocessLabels(config),
            search_output=beam.pvalue.AsSingleton(search_output),
        )
        postprocess_labels | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            "mapped_audio",
            dataset=config.general.dataset_id,
            project=config.general.project,
            schema=schema,
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            method=beam.io.WriteToBigQuery.Method.FILE_LOADS
        )

        # Store results
        audio_output        | "Store Audio (temp)"      >> beam.ParDo(WriteAudio())
        sifted_audio        | "Store Sifted Audio"      >> beam.ParDo(WriteSiftedAudio("butterworth"))
        classifications     | "Store Classifications"   >> beam.ParDo(WriteClassifications(config))
        # postprocess_labels  | "Write Results"           >> beam.io.WriteToText("data/output.txt", shard_name_template="")

        # Output results
        # postprocessed_labels | "Write Results" >> beam.io.WriteToText("output.txt")


if __name__ == "__main__":
    run()
