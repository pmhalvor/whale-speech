from apache_beam.dataframe.convert import to_pcollection
from apache_beam.io import filesystems
from config import load_pipeline_config
from datetime import datetime
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
        export_file = self._get_export_file(start, end)

        species = config.search.species

        geometry_search(geometry_file, start,end, export_file, species)

        results = self._postprocecss(export_file)

        for row in results.to_dict(orient='records'):
            yield row


    def _preprocess_date(self, date_str):
        # return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        return date_str.split("T")[0]


    def _get_geometry_file(self):
        """
        Uses io.Bytes with filesystems.FileSystems.open(data_path)
        to load the geometry file.
        """
        filename = config.search.filename
        geometry_filename = config.search.geometery_file_path_template.format(
            # root_dir=os.path.join("..", "..", config.root_dir), 
            filename=filename
        )
        return io.BytesIO(filesystems.FileSystems.open(geometry_filename).read())
    
    
    def _get_export_file(self, start, end):
        filename = config.search.filename

        export_filename = config.search.export_template.format(
            filename=filename,
            timeframe=(
                f"{start}-{end}"
                if start != end
                else start
            ),
            root_dir=config.root_dir
        )

        return export_filename


    def _postprocecss(self, export_file):
        results = pd.read_csv(export_file)

        results = results[config.search.columns]

        logging.info(f"Results: \n{results.head()}")

        return results
