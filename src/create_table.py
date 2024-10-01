from google.cloud import bigquery
from google.api_core.exceptions import Conflict

from config import load_pipeline_config

config = load_pipeline_config()

client = bigquery.Client()


# Define the table schema
schema = [
    bigquery.SchemaField("key", "STRING"),
    bigquery.SchemaField("audio", "FLOAT64", mode="REPEATED"),  # 'REPEATED' for arrays
    bigquery.SchemaField("pooled_score", "FLOAT64"),
    bigquery.SchemaField("encounter_ids", "STRING", mode="REPEATED"),
    bigquery.SchemaField("encounter_img_urls", "STRING", mode="REPEATED"),
    bigquery.SchemaField("longitude", "FLOAT64"),
    bigquery.SchemaField("latitude", "FLOAT64"),
    bigquery.SchemaField("start", "TIMESTAMP"),
    bigquery.SchemaField("end", "TIMESTAMP"),
]


# Create a dataset
def create_dataset(dataset_id):
    try:
        dataset_path = f"{client.project}.{dataset_id}"
        dataset = bigquery.Dataset(dataset_path)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {client.project}.{dataset.dataset_id}")
    except Conflict as e:
        if "Already Exists" in str(e):
            dataset = client.get_dataset(dataset_id)
            print(f"Dataset {client.project}.{dataset.dataset_id} already exists. Continuing.")
        else:
            raise e
    
    return dataset
        

# Create a table
def create_table(dataset_id, table_id, schema=schema):
    try:
        table_path = f"{client.project}.{dataset_id}.{table_id}"
        table = bigquery.Table(table_path, schema=schema)
        table = client.create_table(table)
        print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
    except Conflict as e:
        if "Already Exists" in str(e):
            table = client.get_table(table_path)
            print(f"Table {table.project}.{table.dataset_id}.{table.table_id} already exists. Continuing.")
        else:
            raise e


def run(args):
    dataset = create_dataset(args.dataset_id)
    table = create_table(dataset.dataset_id, args.table_id)
    return table


    
if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, help="BigQuery dataset ID", default=config.general.dataset_id)
    parser.add_argument("--table_id", type=str, help="BigQuery table ID", default=config.postprocess.postprocess_table_id)

    args = parser.parse_args()

    run(args)
