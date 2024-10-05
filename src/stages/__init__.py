import apache_beam as beam

write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
method=beam.io.WriteToBigQuery.Method.FILE_LOADS
custom_gcs_temp_location="gs://bioacoustics/whale-speech/temp"