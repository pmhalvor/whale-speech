import apache_beam as beam
import logging
import numpy as np
import pandas as pd

# from google.cloud import bigquery
from apache_beam.io.gcp.internal.clients import bigquery

from typing import Dict, Any, Tuple
from types import SimpleNamespace


# class PostprocessLabels(beam.PTransform):
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


        self._init_big_query_writer(config)

    def _init_big_query_writer(self, config: SimpleNamespace):
        self.table_spec = bigquery.TableReference(
            projectId=self.project,
            datasetId=self.dataset_id,
            tableId=self.table_id
        )



    def process(self, element, search_output):
        joined_df = self._build_dfs(element, search_output)

        for row in joined_df.to_dict(orient="records"):
            if row["classifications"] == []:
                logging.info(f"Skipping row with no classification: {row.keys()}")
                logging.info(f"Row: {row}")
                continue

            logging.info(f"Writing row to BigQuery: {row.keys()} \n{row}")
            yield row


    # def expand(self, pcoll, search_output):
    #     return (
    #         pcoll
    #         | "Process" >> beam.ParDo(self._build_dfs, search_output)
    #         | "Write to BigQuery" >> beam.io.WriteToBigQuery(
    #             self.table_spec,
    #             schema=self.schema,
    #             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
    #             create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
    #         )
    #     )


        # for row in self._build_dfs(element, search_output):
        #     row | beam.io.WriteToBigQuery(
        #         self.table_spec,
        #         schema=self.schema,
        #         write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        #         create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
        #     )


    # def _process(self, element: Dict[str, Any], search_output: Dict[str, Any]):
    #     # TODO remove _process (replaced by expand)
    #     logging.info(f"element \n{element}")
    #     logging.info(f"search_output \n{search_output}")

    #     # convert element to dataframe
    #     classifications_df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
    #     classifications_df = classifications_df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
    #     classifications_df["encounter_id"] = classifications_df["encounter_id"].astype(str)
        
    #     # pool classifications in postprocessing
    #     classifications_df["pooled_score"] = classifications_df["classifications"].apply(self._pool_classifications)

    #     # convert search_output to dataframe
    #     search_output = search_output.rename(columns={"id": "encounter_id"})
    #     search_output["encounter_id"] = search_output["encounter_id"].astype(str)  # TODO do in one line
    #     search_output = search_output[[
    #         # TODO refactor to confing
    #         "encounter_id",
    #         "latitude",
    #         "longitude",
    #         "displayImgUrl",
    #         # "species",  # TODO add in geo search stage (require rm local file)
    #     ]]

    #     # join dataframes
    #     joined_df = pd.merge(search_output, classifications_df, how="inner", on="encounter_id")

    #     logging.info(f"Final output: \n{joined_df.head()}")

    #     # write to BigQuery
    #     # self._write_to_bigquery(joined_df)

    #     return joined_df.to_dict(orient="records")

    def _build_dfs(self, element, search_output):
        # logging.info(f"element \n{element}")
        # logging.info(f"search_output \n{search_output}")

        # convert element to dataframe
        classifications_df = self._build_classification_df(element)

        # convert search_output to dataframe
        search_output_df = self._build_search_output_df(search_output)

        # join dataframes
        joined_df = pd.merge(search_output_df, classifications_df, how="inner", on="encounter_id")

        logging.info(f"Final output: \n{joined_df.head()}")
        logging.info(f"Final output columns: {joined_df.columns}")

        return joined_df


    def _build_classification_df(self, element: Tuple) -> pd.DataFrame:
        # convert element to dataframe
        classifications_df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
        classifications_df = classifications_df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
        classifications_df["encounter_id"] = classifications_df["encounter_id"].astype(str)

        # convert audio arrays to list(floats)
        # classifications_df["audio"] = classifications_df["audio"].apply(lambda x: x.tolist())
        # drop audio column
        classifications_df = classifications_df.drop(columns=["audio"])

        # extract classifications from shape (n, 1) to (n,)
        # classifications_df["classifications"] = classifications_df["classifications"].apply(lambda x: x.flatten())
        classifications_df["classifications"] = classifications_df["classifications"].apply(lambda x:[e[0] for e in x])

        # pool classifications in postprocessing
        logging.info(f"Classifications: \n{classifications_df['classifications']}")
        classifications_df["pooled_score"] = classifications_df["classifications"].apply(self._pool_classifications)

        # convert start adn end to isoformat
        classifications_df["start"] = classifications_df["start"].apply(lambda x: x.isoformat())
        classifications_df["end"] = classifications_df["end"].apply(lambda x: x.isoformat())

        logging.info(f"Classifications: \n{classifications_df.head()}")
        logging.info(f"Classifications shape: {classifications_df.shape}")

        return classifications_df


    def _build_search_output_df(self, search_output: Dict[str, Any]) -> pd.DataFrame:
        # convert search_output to dataframe
        search_output = search_output.rename(columns={"id": "encounter_id"})
        search_output["encounter_id"] = search_output["encounter_id"].astype(str)
        search_output = search_output[[
            # TODO refactor to confing
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
    