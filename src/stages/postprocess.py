import apache_beam as beam
import logging
import numpy as np
import pandas as pd
import os 

from apache_beam.io import filesystems

from typing import Dict, Any, Tuple
from types import SimpleNamespace


class PostprocessLabels(beam.DoFn):
    def __init__(self, config: SimpleNamespace):
        self.config = config

        # self.search_output_path_template = config.search.output_path_template
        # self.sifted_audio_path_template = config.sift.output_array_path_template
        # self.classification_path = config.classify.output_array_path_template

        self.pooling = config.postprocess.pooling
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.postprocess.postprocess_table_id

        # storage params
        self.is_local = config.general.is_local
        self.output_path = config.postprocess.output_path
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.postprocess.postprocess_table_id
        self.columns = list(vars(config.postprocess.postprocess_table_schema))
        self.schema = self._schema_to_dict(config.postprocess.postprocess_table_schema)


    @staticmethod
    def _schema_to_dict(schema):
        return {
            "fields": [
                {
                    "name": name, 
                    "type": getattr(schema, name).type, 
                    "mode": getattr(schema, name).mode
                } 
                for name in vars(schema)
            ]
        }

    def process(self, element, search_output):
        # convert element to dataframe
        classifications_df = self._build_classification_df(element)

        # clean up search_output dataframe
        search_output_df = self._build_search_output_df(search_output)

        # join dataframes
        joined_df = pd.merge(search_output_df, classifications_df, how="inner", on="encounter_id")

        # add paths 
        final_df = self._add_paths(joined_df)

        logging.info(f"Final output: \n{final_df.head()}")
        logging.info(f"Final output columns: {final_df.columns}")

        self._store(final_df)       

        yield final_df.to_dict(orient="records")

    def _build_classification_df(self, element: Tuple) -> pd.DataFrame:
        # convert element to dataframe
        df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
        df = df[df["classifications"].apply(lambda x: len(x) > 0)]  # rm empty rows

        # explode encounter_ids
        df = df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
        df["encounter_id"] = df["encounter_id"].astype(str)

        # pool classifications in postprocessing
        df["pooled_score"] = df["classifications"].apply(self._pool_classifications)

        # convert start and end to isoformat
        df["start"] = df["start"].apply(lambda x: x.isoformat())
        df["end"] = df["end"].apply(lambda x: x.isoformat())

        # drop audio and classification columns 
        df = df.drop(columns=["audio"])
        df = df.drop(columns=["classifications"])

        logging.info(f"Classifications: \n{df.head()}")
        logging.info(f"Classifications shape: {df.shape}")
        return df.reset_index(drop=True)

    def _build_search_output_df(self, search_output: Dict[str, Any]) -> pd.DataFrame:
        # convert search_output to dataframe
        search_output["encounter_id"] = search_output["id"].astype(str)
        search_output = search_output[[
            "encounter_id",
            "latitude",
            "longitude",
            "displayImgUrl",
            # "species",  # TODO add in geo search stage (require rm local file)
        ]]
        logging.info(f"Search output: \n{search_output.head()}")
        logging.info(f"Search output shape: {search_output.shape}")

        return search_output

    def _pool_classifications(self, classifications: np.array) -> Dict[str, Any]:
        if self.pooling == "mean" or self.pooling == "avg" or self.pooling == "average":
            pooled_score = np.mean(classifications)
        elif self.pooling == "max":
            pooled_score = np.max(classifications)
        elif self.pooling == "min":
            pooled_score = np.min(classifications)
        else:
            raise ValueError(f"Pooling method {self.pooling} not supported.")
        
        return pooled_score

    def _add_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        df["audio_path"] = "NotImplemented"
        df["classification_path"] = "NotImplemented"
        df["img_path"] = df["displayImgUrl"] 
        df = df.drop(columns=["displayImgUrl"])
        return df

    def _store(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        
        if self.is_local:
            self._store_local(df)
        else:
            self._store_bigquery(df)
    
    def _store_bigquery(self, element):
        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
        method=beam.io.WriteToBigQuery.Method.FILE_LOADS
        custom_gcs_temp_location="gs://bioacoustics/whale-speech/temp"

        logging.info(f"Writing to BigQuery")
        logging.info(f"Table: {self.table_id}")
        logging.info(f"Dataset: {self.dataset_id}")
        logging.info(f"Project: {self.project}")
        logging.info(f"Schema: {self.schema}")
        logging.info(f"Len element:  {len(element)}")
        logging.info(f"Element keys: {element[0].keys()}")
        
        element | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            self.table_id,
            dataset=self.dataset_id,
            project=self.project,
            # "bioacoustics-2024.whale_speech.mapped_audio",
            schema=self.schema,
            write_disposition=write_disposition,
            create_disposition=create_disposition,
            method=method,
            custom_gcs_temp_location=custom_gcs_temp_location
        )

        yield element

    def _store_local(self, element):
        logging.info(f"Storing to local path: {self.output_path}")

        if filesystems.FileSystems.exists(self.output_path):
            stored_df = pd.read_json(self.output_path, orient="records")

            # convert encounter_id to str
            stored_df["encounter_id"] = stored_df["encounter_id"].astype(str)

        else:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            stored_df = pd.DataFrame([], columns=self.columns)
        
        element_df = pd.DataFrame(element, columns=self.columns)
        final_df = pd.concat([stored_df, element_df], ignore_index=True)
        final_df = final_df.drop_duplicates()
        logging.debug(f"Appending df to {self.output_path} \n{final_df}")

        # store as json (hack: to remove \/\/ escapes)
        final_df_json = final_df.to_json(orient="records").replace("\\/", "/")
        print(final_df_json, file=open(self.output_path, "w"))
