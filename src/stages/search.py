from apache_beam.io import filesystems
from happywhale.happywhale import geometry_search
from types import SimpleNamespace

import apache_beam as beam
from apache_beam.utils.timestamp import Timestamp
from datetime import timedelta, timezone
import io 
import logging
import os
import pandas as pd
import pytz

logger = logging.getLogger(__name__)



class GeometrySearch(beam.DoFn):

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.filesystem = config.general.filesystem.lower()

        self.species = config.search.species

        self.filename = config.search.filename
        self.geometry_file_path_template = config.search.geometry_file_path_template
        self.search_columns = config.search.search_columns

        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.search.search_table_id
        self.schema = self._schema_to_dict(config.search.search_table_schema)
        self.temp_location = config.general.temp_location
        self.write_params = config.bigquery.__dict__

        self.output_path = config.search.output_path_template.format(
            table_id=self.table_id,
            geofile=self.filename
        )


    def process(self, element):
        start = self._preprocess_date(element.get('start'))
        end = self._preprocess_date(element.get('end'))
        
        geometry_file = self._get_geometry_file()
        export_file = self._get_file_buffer("csv")


        logger.info(f"Searching for {self.species} between {start} and {end} in {geometry_file}.")
        geometry_search(geometry_file, start,end, export_file, self.species)

        search_results = self._postprocess(export_file)

        self._store(search_results)

        yield search_results


    @staticmethod
    def _preprocess_date(date_str):
        return date_str.split("T")[0] if "T" in date_str else date_str


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
        results = results.dropna() # NOTE: can be removed if encounter_time column type updated
        logging.info(f"Search results: \n{results.head()}")
        return results


    def _store(self, search_results):
        rows = self._convert_to_table_rows(search_results)

        if self.filesystem == "local":
            # convert back to dataframe w7 correct columns 
            search_results = pd.DataFrame(rows)
            if not filesystems.FileSystems.exists(self.output_path):
                if not filesystems.FileSystems.exists(os.path.dirname(self.output_path)):
                    # safely create parent dir (in case something gets wrongly deleted)
                    filesystems.FileSystems.mkdirs(os.path.dirname(self.output_path))
                search_results.to_csv(self.output_path, index=False)
            else:
                previous_search_results = pd.read_csv(self.output_path)
                search_results = pd.concat([previous_search_results, search_results])
                search_results.drop_duplicates(inplace=True)
                search_results.to_csv(self.output_path, index=False)

        elif self.filesystem == "gcp":
            logging.info(f"search_results.columns: {search_results.columns}")

            rows | f"Update {self.table_id}" >> beam.io.WriteToBigQuery(
                self.table_id,
                dataset=self.dataset_id,
                project=self.project,
                schema=self.schema,
                custom_gcs_temp_location=self.temp_location,
                **self.write_params
            )
        else:
            raise ValueError(f"Filesystem {self.filesystem} not supported.")

        logging.info(f"Stored search results in {self.output_path}")


    def _convert_to_table_rows(self, df):
        table_colums = [field["name"] for field in self.schema["fields"]]
        
        df["encounter_id"] = df["id"]
        df["img_path"] = df["displayImgUrl"]
        df["longitude"] = df["longitude"].astype(float)
        df["latitude"] = df["latitude"].astype(float)

        df["encounter_time"] = df.apply(self._apply_timezone, axis=1)

        df = df[[*table_colums]]

        return df.to_dict(orient="records")
        
    @staticmethod
    def _apply_timezone(row):
        dt_str = f"{row['startDate']} {row['startTime']}"
        dt = pd.to_datetime(dt_str)
        try:
            timezone_str = row['timezone']
            if timezone_str.startswith(('-', '+')):
                # NOTE: store in UTC for global consistency
                hours_offset = int(timezone_str[:3])
                minutes_offset = int(timezone_str[4:]) if len(timezone_str) > 3 else 0
                tz_offset = timezone(timedelta(hours=hours_offset, minutes=minutes_offset))
                dt = dt.replace(tzinfo=tz_offset)

            else:
                dt = dt.tz_localize(pytz.timezone(timezone_str))
        except Exception as e:
            raise e
        
        return dt.astimezone(pytz.UTC).isoformat()