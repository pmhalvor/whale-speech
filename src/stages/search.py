from apache_beam.io import filesystems
from happywhale.happywhale import geometry_search
from types import SimpleNamespace

import apache_beam as beam
import io 
import logging
import os
import pandas as pd


class GeometrySearch(beam.DoFn):

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.is_local = config.general.is_local

        self.species = config.search.species

        self.filename = config.search.filename
        self.geometry_file_path_template = config.search.geometry_file_path_template
        self.search_columns = config.search.search_columns

        self.output_path = config.search.output_path_template.format(geofile=self.filename)
        
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.search.search_table_id
        self.schema = self._schema_to_dict(config.search.search_table_schema)
        self.temp_location = config.general.temp_location
        self.write_params = config.bigquery.__dict__


    def process(self, element):
        start = self._preprocess_date(element.get('start'))
        end = self._preprocess_date(element.get('end'))
        
        geometry_file = self._get_geometry_file()
        export_file = self._get_file_buffer("csv")

        geometry_search(geometry_file, start,end, export_file, self.species)

        search_results = self._postprocess(export_file)

        self._store(search_results)

        yield search_results


    @staticmethod
    def _preprocess_date(date_str):
        return date_str.split("T")[0]


    @staticmethod
    def _get_file_buffer(filetype="csv"):
        fb = io.BytesIO()  # or io.StringIO()
        fb.endswith = lambda x: x=="csv"  # hack to ensure happywhale saves df to fb
        return fb
    

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


    def _get_geometry_file(self):
        """
        Uses io.Bytes with filesystems.FileSystems.open(data_path)
        to load the geometry file.
        """
        filename = self.filename
        geometry_filename = self.geometry_file_path_template.format(
            filename=filename
        )
        return io.BytesIO(filesystems.FileSystems.open(geometry_filename).read())
    

    def _postprocess(self, export_file) -> pd.DataFrame:
        if isinstance(export_file, io.BytesIO):
            export_file.seek(0) 
        results = pd.read_csv(export_file)
        results = results[self.search_columns]
        logging.info(f"Search results: \n{results.head()}")
        return results


    def _store(self, search_results):

        if self.is_local:
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                search_results.to_csv(self.output_path, index=False)
            else:
                previous_search_results = pd.read_csv(self.output_path)
                search_results = pd.concat([previous_search_results, search_results])
                search_results.drop_duplicates(inplace=True)
                search_results.to_csv(self.output_path, index=False)

        else:
            logging.info(f"search_results.columns: {search_results.columns}")

            # write to bigquery
            rows = self._convert_to_table_rows(search_results)
            
            rows | f"Update {self.table_id}" >> beam.io.WriteToBigQuery(
                self.table_id,
                dataset=self.dataset_id,
                project=self.project,
                # "bioacoustics-2024.whale_speech.mapped_audio",
                schema=self.schema,
                custom_gcs_temp_location=self.temp_location,
                **self.write_params
            )

        logging.info(f"Stored search results in {self.output_path}")


    def _convert_to_table_rows(self, df):
        table_colums = [field["name"] for field in self.schema["fields"]]
        
        df["encounter_id"] = df["id"]
        df["img_path"] = df["displayImgUrl"]
        df["longitude"] = df["longitude"].astype(float)
        df["latitude"] = df["latitude"].astype(float)

        df["encounter_time"] = df[["startDate", "startTime"]].apply(
            lambda x: f"{x.startDate}T{x.startTime}", axis=1
        )

        df = df[[*table_colums]]

        return df.to_dict(orient="records")
