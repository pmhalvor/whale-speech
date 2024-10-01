import apache_beam as beam
import logging
import numpy as np
import pandas as pd

# from google.cloud import bigquery
from apache_beam.io.gcp.internal.clients import bigquery

from typing import Dict, Any, Tuple
from types import SimpleNamespace


class PostprocessLabels(beam.DoFn):
    def __init__(self, config: SimpleNamespace):
        self.config = config

        self.search_output_path_template = config.search.export_template
        self.sifted_audio_path_template = config.sift.output_path_template
        self.classification_path = config.classify.classification_path

        self.pooling = config.postprocess.pooling
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.postprocess.postprocess_table_id

        self.client = bigquery.Client() # requires creatials set up in env
        self.table_ref = f"{self.project}.{self.dataset_id}.{self.table_id}"


    def process(self, element: Dict[str, Any], search_output: Dict[str, Any]):
        logging.info(f"element \n{element}")
        logging.info(f"search_output \n{search_output}")

        # convert element to dataframe
        classifications_df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
        classifications_df = classifications_df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
        classifications_df["encounter_id"] = classifications_df["encounter_id"].astype(str)
        
        # pool classifications in postprocessing
        classifications_df["pooled_score"] = classifications_df["classifications"].apply(self._pool_classifications)

        # convert search_output to dataframe
        search_output = search_output.rename(columns={"id": "encounter_id"})
        search_output["encounter_id"] = search_output["encounter_id"].astype(str)  # TODO do in one line
        search_output = search_output[[
            # TODO refactor to confing
            "encounter_id",
            "latitude",
            "longitude",
            "displayImgUrl",
            # "species",  # TODO add in geo search stage (require rm local file)
        ]]

        # join dataframes
        joined_df = pd.merge(search_output, classifications_df, how="inner", on="encounter_id")

        logging.info(f"Final output: \n{joined_df.head()}")

        # write to BigQuery
        # self._write_to_bigquery(joined_df)

        return joined_df.to_dict(orient="records")

    def _build_classification_df(self, element: Tuple) -> pd.DataFrame:
        # convert element to dataframe
        classifications_df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
        classifications_df = classifications_df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
        classifications_df["encounter_id"] = classifications_df["encounter_id"].astype(str)

        # convert audio arrays to list(floats)
        classifications_df["audio"] = classifications_df["audio"].apply(lambda x: x.tolist())


        # pool classifications in postprocessing
        # TODO check that shape (n,1) is handled correctly
        classifications_df["pooled_score"] = classifications_df["classifications"].apply(self._pool_classifications)
        logging.info(f"Classifications: \n{classifications_df.head()}")
        logging.info(f"Classifications shape: {classifications_df.shape}")
        

        return classifications_df


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
    

    def _write_to_bigquery(self, df: pd.DataFrame):

        for row in df.to_dict(orient="records"):
            self._insert_row(row)
            logging.debug(f"Inserted row {row} to BigQuery table {self.table_ref}")


    def _insert_row(self, row: Dict[str, Any]):
        # Insert data into BigQuery
        errors = self.client.insert_rows_json(self.table_ref, [row])
        if errors:
            raise Exception(f"Error inserting rows: {errors}")
