from apache_beam.io import filesystems
from config import load_pipeline_config
from happywhale.happywhale import geometry_search

import apache_beam as beam
import io 
import logging
import os
import pandas as pd


config = load_pipeline_config()


class GeometrySearch(beam.DoFn):

    def process(self, element):
        start = self._preprocess_date(element.get('start'))
        end = self._preprocess_date(element.get('end'))
        
        geometry_file = self._get_geometry_file()
        export_file = self._get_export_path(start, end)

        species = config.search.species

        geometry_search(geometry_file, start,end, export_file, species)

        yield self._postprocess(export_file)


    def _preprocess_date(self, date_str):
        # return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return date_str.split("T")[0]


    def _get_geometry_file(self):
        """
        Uses io.Bytes with filesystems.FileSystems.open(data_path)
        to load the geometry file.
        """
        filename = config.search.filename
        geometry_filename = config.search.geometry_file_path_template.format(
            filename=filename
        )
        return io.BytesIO(filesystems.FileSystems.open(geometry_filename).read())
    
    
    def _get_export_path(self, start, end):
        filename = config.search.filename

        export_filename = config.search.export_template.format(
            filename=filename,
            timeframe=(
                f"{start}-{end}"
                if start != end
                else start
            )
        )

        return export_filename


    def _postprocess(self, export_file) -> pd.DataFrame:
        results = pd.read_csv(export_file)

        results = results[config.search.columns]

        logging.info(f"Search results: \n{results.head()}")

        return results
