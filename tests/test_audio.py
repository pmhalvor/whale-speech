from datetime import datetime
from types import SimpleNamespace

import pytest
import pandas as pd
import numpy as np

from stages.audio import RetrieveAudio

@pytest.fixture
def config():
    return SimpleNamespace(
        audio=SimpleNamespace(
            margin = 300,
            offset = 1,
            source_sample_rate = 16_000,
            url_template = "https://pacific-sound-16khz.s3.amazonaws.com/{year}/{month:02}/{filename}",
            filename_template = "MARS-{year}{month:02}{day:02}T000000Z-16kHz.wav",
            output_array_path_template = "output_array_path_template",
            output_table_path_template = "output_table_path_template",
            skip_existing = True,

            store_audio = False,
            audio_table_id = "config.audio.audio_table_id",
            audio_table_schema = SimpleNamespace(key=SimpleNamespace(type="STRING", mode="REQUIRED"))

        ),
        general=SimpleNamespace(
            debug = True,
            filesystem = "local",
            project="project", 
            dataset_id="dataset_id",
            workbucket = "workbucket",
            temp_location="temp_location",
        ),
        bigquery = SimpleNamespace(write_disposition="write_disposition"), 
    )

@pytest.fixture
def sample_search_results_df():
    data = {
        'id':           [1, 2, 3],
        'startDate':    ['2024-07-08', '2024-09-10', '2024-09-10'],
        'startTime':    ['00:13:00.123456', '01:37:00', '01:32:00'],
        'endTime':      ['00:13:00', '01:37:00', '01:42:00.133700']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_time_frame_df():
    data = {
        'id':           [1, 2, 3],
        'startDate':    ['2024-07-08', '2024-09-10', '2024-09-10'],
        'startTime':    [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 32, 0)],
        'endTime':      [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 42, 0)],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 32, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 42, 0),  datetime(2024, 9, 10, 0, 47, 0)]
    }
    return pd.DataFrame(data)


def test_build_time_frames(config, sample_search_results_df):
    # Assemble
    expected_df = pd.DataFrame({
        'id':           [1, 2, 3],
        'startDate':    ['2024-07-08', '2024-09-10', '2024-09-10'],
        'startTime':    [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 32, 0)],
        'endTime':      [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 42, 0)],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 32, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 42, 0),  datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio(config)._build_time_frames(sample_search_results_df)
    
    # Assert
    assert pd.api.types.is_datetime64_any_dtype(actual_df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(actual_df['end_time'])
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_find_overlapping(config, sample_time_frame_df):
    # Assemble
    expected_df = pd.DataFrame({
        'encounter_ids':[["1"], ["3", "2"]],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio(config)._find_overlapping(sample_time_frame_df)
    
    # Assert
    pd.testing.assert_frame_equal(expected_df, actual_df)
    assert pd.api.types.is_datetime64_any_dtype(actual_df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(actual_df['end_time'])


def test_preprocess(config, sample_search_results_df):
    # Assemble
    expected_df = pd.DataFrame({
        'encounter_ids':[["1"], ["3", "2"]],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio(config)._preprocess(sample_search_results_df)
    
    # Assert
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_get_file_url(config):
    # Assemble
    sample_year = 2024
    sample_month = 9
    sample_day = 10
    expected_url = 'https://pacific-sound-16khz.s3.amazonaws.com/2024/09/MARS-20240910T000000Z-16kHz.wav'

    # Act
    actual_url = RetrieveAudio(config)._get_file_url(sample_year, sample_month, sample_day)
    
    # Assert
    assert expected_url == actual_url

# NOTE: skipping tests for _load and process, since they require downloading the audio file
