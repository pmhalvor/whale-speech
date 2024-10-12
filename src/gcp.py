from google.cloud import bigquery
from google.api_core.exceptions import Conflict

from config import load_pipeline_config

import logging

logging.basicConfig(level=logging.INFO)


config = load_pipeline_config()

client = bigquery.Client()


def create_dataset(dataset_id):
    try:
        dataset_path = f"{client.project}.{dataset_id}"
        dataset = bigquery.Dataset(dataset_path)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        logging.info(f"Created dataset {client.project}.{dataset.dataset_id}")
    except Conflict as e:
        if "Already Exists" in str(e):
            dataset = client.get_dataset(dataset_id)
            logging.info(f"Dataset {client.project}.{dataset.dataset_id} already exists. Continuing.")
        else:
            raise e
    
    return dataset


def table_exists(table_id: str):
    return table_id in [table.table_id for table in client.list_tables(config.general.dataset_id)]


def get_partition_columns(table_id: str):
    columns =  [
        field.name 
        for field in client.get_table(
            f"{config.general.project}.{config.general.dataset_id}.{table_id}"
        ).schema
        if field.name in config.general.partition_columns
    ]
    identifier = "key" if "key" in columns else "encounter_id"
    
    return ",".join(columns), identifier


def deduplicate_table(table_id: str):
    if not table_exists(table_id):
        logging.info(f"Table {table_id} does not exist. Skipping deduplication.")
        return

    columns, identifier = get_partition_columns(table_id)

    query = f"""
    CREATE OR REPLACE TABLE `{config.general.project}.{config.general.dataset_id}.{table_id}` AS
    SELECT *
    FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY {columns} ORDER BY {identifier} DESC) AS row_num
    FROM `{config.general.project}.{config.general.dataset_id}.{table_id}`
    )
    WHERE row_num = 1
    """
    logging.info(f"Running deduplicatation query: \n {query}")
    client.query(query)
    logging.info(f"Deduplicated table {table_id}.")


def initialize():
    dataset = create_dataset(config.general.dataset_id)
    return dataset


def deduplicate():
    for table_id in config.general.tables:
        deduplicate_table(table_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GCP Utility Functions")
    parser.add_argument("--init", action="store_true", help="Initialize BigQuery dataset (config.general.dataset_id)")
    parser.add_argument("--deduplicate", action="store_true", help="Dedupliacte BigQuery tables (config.general.tables)")
    args = parser.parse_args()

    if args.init:
        initialize()
    elif args.deduplicate:
        deduplicate()
    else:
        parser.print_help()
