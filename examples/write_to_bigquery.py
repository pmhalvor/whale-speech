
from apache_beam.io.gcp.internal.clients import bigquery
import apache_beam as beam


# project-id:dataset_id.table_id
table_spec = 'bioacoustics-2024.whale_speech.sample_quotes'


table_schema = {
    'fields': [{
        'name': 'source', 'type': 'STRING', 'mode': 'NULLABLE'
    }, {
        'name': 'quote', 'type': 'STRING', 'mode': 'REQUIRED'
    }]
}

# Create a pipeline
temp_location = "gs://bioacoustics/whale-speech/temp"

with beam.Pipeline(options=beam.options.pipeline_options.PipelineOptions()) as pipeline:

    quotes = pipeline | beam.Create([
        {
            'source': 'Mahatma Gandhi', 'quote': 'My life is my message.'
        },
        {
            'source': 'Yoda', 'quote': "Do, or do not. There is no 'try'."
        },
    ])
    quotes | beam.io.WriteToBigQuery(
        table_spec,
        schema=table_schema,
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        custom_gcs_temp_location=temp_location,
        method=beam.io.WriteToBigQuery.Method.FILE_LOADS
    )

print("Completed writing to BigQuery")