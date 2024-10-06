from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from stages.search import GeometrySearch
from types import SimpleNamespace
from unittest.mock import patch

@pytest.fixture
def config():
    return SimpleNamespace(
        search=SimpleNamespace(
            output_path_template="template",
            species="species",
            filename="filename",
            geometry_file_path_template="geometry_file_path_template",
            search_columns=[
                "id",
                "latitude",
                "longitude",
                "startDate",
                "startTime",
                "endTime",
                "timezone",
                "displayImgUrl",
            ],
            search_table_id="search_table_id",
            search_table_schema=SimpleNamespace(
                encounter_id=SimpleNamespace(type="STRING", mode="REQUIRED"),
                encounter_time=SimpleNamespace(type="STRING", mode="REQUIRED"),
                latitude=SimpleNamespace(type="FLAOT64", mode="REQUIRED"),
                longitude=SimpleNamespace(type="FLOAT64", mode="REQUIRED"),
                img_path=SimpleNamespace(type="STRING", mode="NULLABLE"),
            )
        ),
        general=SimpleNamespace(
            is_local=True,
            project="project", 
            dataset_id="dataset_id",
            temp_location="temp_location"
        ),
        bigquery=SimpleNamespace(
            write_disposition="write_disposition",
            create_disposition="create_disposition",
            method="method",
            custom_gcs_temp_location="custom_gcs_temp_location"
        )
    )


@pytest.fixture
def sample_element():
    return {
        'start': '2024-07-08T00:13:00',
        'end': '2024-07-08T00:13:00'
    }


def test_preprocess_date(sample_element, config):
    # Assemble
    expected = '2024-07-08'

    # Act
    actual = GeometrySearch(config)._preprocess_date(sample_element.get('start'))

    # Assert
    assert actual == expected


def test_postprocess(sample_element, config):
    # Assemble
    sample_export_file = "tests/data/sample_2024-07-08.csv"
    expected = pd.DataFrame({
        "id": [471768],
        "latitude": [36.727764],
        "longitude": [-122.068258],
        "startDate": ["2024-07-08"],
        "startTime": ["15:17:52"],
        "endTime": ["15:17:52"],
        "timezone": ["-07:00"],
        "displayImgUrl": ["https://au-hw-media-m.happywhale.com/c9fe66f6-f55e-40aa-9c50-c31f3fa735a4.jpg"],
    })

    # Act
    actual = GeometrySearch(config)._postprocess(sample_export_file)

    # Assert
    pd.testing.assert_frame_equal(expected, actual)
