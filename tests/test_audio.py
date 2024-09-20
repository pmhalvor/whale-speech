from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from stages.audio import RetrieveAudio
from unittest.mock import patch


@pytest.fixture
def sample_search_results_df():
    data = {
        'id':           [1, 2, 3],
        'startDate':    ['2024-07-08', '2024-09-10', '2024-09-10'],
        'startTime':    ['00:13:00', '01:37:00', '01:32:00'],
        'endTime':      ['00:13:00', '01:37:00', '01:42:00']
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


@patch('stages.audio.config')
def test_build_time_frames(mock_config, sample_search_results_df):
    # Assemble
    # mock config values to avoid failing tests on config changes
    mock_config.audio.margin = 5 * 60 # 5 minutes
    mock_config.audio.offset = 1  # 1 hour # TODO remove. only used for development

    expected_df = pd.DataFrame({
        'id':           [1, 2, 3],
        'startDate':    ['2024-07-08', '2024-09-10', '2024-09-10'],
        'startTime':    [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 32, 0)],
        'endTime':      [datetime(2024, 7, 8, 0, 13, 0),  datetime(2024, 9, 10, 1, 37, 0),  datetime(2024, 9, 10, 1, 42, 0)],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 32, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 42, 0),  datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio()._build_time_frames(sample_search_results_df)
    
    # Assert
    assert pd.api.types.is_datetime64_any_dtype(actual_df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(actual_df['end_time'])
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_find_overlapping(sample_time_frame_df):
    # Assemble
    expected_df = pd.DataFrame({
        'encounter_ids':[["1"], ["3", "2"]],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio()._find_overlapping(sample_time_frame_df)
    
    # Assert
    pd.testing.assert_frame_equal(expected_df, actual_df)
    assert pd.api.types.is_datetime64_any_dtype(actual_df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(actual_df['end_time'])


@patch('stages.audio.config')
def test_preprocess(mock_config, sample_search_results_df):
    # Assemble
    # mock config values to avoid failing tests on config changes
    mock_config.audio.margin = 5 * 60   # 5 minutes
    mock_config.audio.offset = 1        # 1 hour # TODO remove. only used for development

    expected_df = pd.DataFrame({
        'encounter_ids':[["1"], ["3", "2"]],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 47, 0)]
    })

    # Act
    actual_df = RetrieveAudio()._preprocess(sample_search_results_df)
    
    # Assert
    pd.testing.assert_frame_equal(expected_df, actual_df)


def test_get_file_url():
    # Assemble
    sample_year = 2024
    sample_month = 9
    sample_day = 10
    expected_url = 'https://pacific-sound-16khz.s3.amazonaws.com/2024/09/MARS-20240910T000000Z-16kHz.wav'

    # Act
    actual_url = RetrieveAudio()._get_file_url(sample_year, sample_month, sample_day)
    
    # Assert
    assert expected_url == actual_url

# NOTE: skipping tests for _load and process, since they require downloading the audio file
