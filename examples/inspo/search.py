from happywhale.happywhale import geometry_search
from datetime import datetime
import geopandas as gpd
import os

import config

search_config = config.load("search")

## Params
geometry_file = search_config.file_template.format(root_dir=config.root_dir)
filename = geometry_file.split('/')[-1].split('.')[0]

species = search_config.species

start_str = search_config.start
end_str = search_config.end
start_time = datetime.strptime(search_config.start, "%Y-%m-%d")    
end_time = datetime.strptime(search_config.end, "%Y-%m-%d")

export_file = search_config.export_template.format(
    filename=filename,
    timeframe=(
        f"{start_time.strftime('%Y%m%d')}-{end_time.strftime('%Y%m%d')}"
        if start_time.date() != end_time.date()
        else start_time.strftime('%Y%m%d')
    ),
    root_dir=config.root_dir
)


def run(
        geometry_file=geometry_file, 
        start=start_str,
        end=end_str,
        export_file=export_file,
        species=species
    ):
    """
    1. Run search on happy for given parameters. 
    2. Store the results in a file. Return 
    3. Return filep ath 
    """
    geometry_search(geometry_file, start, end, export_file, species)
    return export_file


def load(export_file=export_file):
    """
    Load the search results from a file.
    """
    if not os.path.exists(export_file):
        raise FileNotFoundError(f"File not found: {export_file}")
    return gpd.read_file(export_file)


if __name__ == "__main__":
    try:
        print(load().head())
    except FileNotFoundError as e:
        print("Run the search first.")
        print(run())
        print(load().head())

