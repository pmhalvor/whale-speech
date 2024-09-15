import happywhale.happywhale as hw

# geometery search for a time-frame
geometry_file = "data/geo/monterey_bay_50kmr.geojson"
start = "2024-09-01"
end = "2024-09-02"
export = "data/encounters/monterey_bay_50kmr_2024-09-01_2024-09-02.geojson"
species = "humpback_whale"

hw.geometry_search(geometry_file, start, end, export, species)
