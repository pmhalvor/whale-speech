from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from stages.search import GeometrySearch
from unittest.mock import patch


@pytest.fixture
def sample_element():
    return {
        'start': '2024-07-08T00:13:00',
        'end': '2024-07-08T00:13:00'
    }


def test_preprocess_date(sample_element):
    # Assemble
    expected = '2024-07-08'

    # Act
    actual = GeometrySearch()._preprocess_date(sample_element.get('start'))

    # Assert
    assert actual == expected


def test_get_export_file(sample_element):
    # Assemble
    start = '2024-07-08'
    end = '2024-07-08'
    expected = "data/encounters/monterey_bay_50km-2024-07-08.csv"

    # Act
    actual = GeometrySearch()._get_export_file(start, end)

    # Assert
    assert actual == expected


def test_postprocess(sample_element):
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
    actual = GeometrySearch()._postprocess(sample_export_file)

    # Assert
    pd.testing.assert_frame_equal(expected, actual)
