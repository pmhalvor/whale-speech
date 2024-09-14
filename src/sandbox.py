import geojson
import geopandas as gpd
import config
import os

search_config = config.load("search")
filename = search_config.file_template.format(root_dir=config.root_dir)

print(filename)
print(os.path.exists(filename))


# with open('data/sightings/monterey_bay_50kmr_2024-09-01_2024-09-02.geojson') as f:
#     data = geojson.load(f)

# df = gpd.GeoDataFrame.from_features(data['features'])
# df = gpd.read_file('data/sightings/monterey_bay_50kmr_2024-09-01_2024-09-02.geojson')

# print(df.head())
# print(df.columns)