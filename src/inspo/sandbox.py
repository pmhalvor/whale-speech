import geojson
import geopandas as gpd
import config
import os

search_config = config.load("search")
filename = search_config.file_template.format(root_dir=config.root_dir)

print(filename)
print(os.path.exists(filename))

