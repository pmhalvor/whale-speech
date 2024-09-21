from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stages.sift import BaseSift, Butterworth
from unittest.mock import patch


@pytest.fixture
def sample_audio_results_df():
    data = {
        'audio':        [np.array([0, 1, 2, 3, 2, 1]),  np.array([1, 2, 3, 2]),             np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*16_500)],
        'start_time':   [datetime(2024, 7, 7, 23, 8, 0),  datetime(2024, 9, 10, 0, 32, 0),  datetime(2024, 9, 10, 0, 27, 0)],
        'end_time':     [datetime(2024, 7, 7, 23, 18, 0), datetime(2024, 9, 10, 0, 42, 0),  datetime(2024, 9, 10, 0, 47, 0)],
        'encounter_ids':[["1"], ["3", "2"], ["2"]]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_audio_results_row():
    audio = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*16_500*60) # bigger than max_duration
    start_time = datetime(2024, 7, 7, 23, 8, 0)
    end_time = datetime(2024, 7, 7, 23, 18, 0)
    encounter_ids = ["encounter1", "encounter2"]
    yield  audio, start_time, end_time, encounter_ids


@pytest.fixture
def sample_batch():
    """
    Cherry picked example data with a sginal that has butterowrth detection w/ params:
    - lowcut = 50
    - highcut = 1500
    - order = 2
    
    """
    signal = np.load("tests/data/20161221T004930-005030-9182.npy")
    key = "20161221T004930-005030-9182"
    return (key, signal)


def test_build_key(sample_audio_results_row):
    # Assemble
    _, start, end, encounter_ids = sample_audio_results_row
    
    expected_key = "20240707T230800-231800_encounter1_encounter2"

    # Act
    actual_key = BaseSift()._build_key(start, end, encounter_ids)

    # Assert
    assert expected_key == actual_key


def test_preprocess(sample_audio_results_row):
    # Assemble
    expected_data = [
        ("20240707T230800-231800_encounter1_encounter2", np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*16_000*60)),
        ("20240707T230800-231800_encounter1_encounter2", np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*500*60)),
    ]

    # Act
    actual_data_generator = Butterworth()._preprocess(sample_audio_results_row)

    # Assert
    for expected in expected_data:
        actual = next(actual_data_generator)  # unload generator

        assert expected[0] == actual[0] # key
        assert expected[1].shape == actual[1].shape # data


def test_postprocess(sample_audio_results_row):
    # Assemble
    pcoll = sample_audio_results_row 
    min_max_detections = {
        "20240707T230800-231800_encounter1_encounter2": {"min":0, "max": 5}  # samples
    }

    expected_data = (
        np.array([0, 1, 2, 3, 4]),          # audio
        datetime(2024, 7, 7, 23, 8, 0),     # start_time
        datetime(2024, 7, 7, 23, 18, 0),    # end_time
        ["encounter1", "encounter2"]        # encounter_ids
    )

    # Act
    actual_data = Butterworth()._postprocess(pcoll, min_max_detections)

    # Assert
    assert len(expected_data) == len(actual_data)
    for expected, actual in zip(expected_data, actual_data):
        assert np.array_equal(expected, actual)


@patch('stages.sift.config')
def test_butter_bandpass(mock_config):
    # Assemble
    mock_config.sift.butterworth.lowcut = 50
    mock_config.sift.butterworth.highcut = 1500
    mock_config.sift.butterworth.order = 2
    mock_config.sift.butterworth.output = "sos"

    expected_coefficients = [
        [ 0.05711683,  0.11423366,  0.05711683,  1.        , -1.22806805,         0.4605427 ],
        [ 1.        , -2.        ,  1.        ,  1.        , -1.97233136,         0.97273604]
    ]

    actual_coefficients = Butterworth()._butter_bandpass()

    # Assert
    assert len(actual_coefficients) == 2  # order
    assert np.allclose(expected_coefficients, actual_coefficients)


@patch('stages.sift.config')
def test_frequency_filter_sift(mock_config, sample_batch):
    # Assemble
    mock_config.sift.butterworth.lowcut = 50
    mock_config.sift.butterworth.highcut = 1500 
    mock_config.sift.butterworth.order = 5
    mock_config.sift.butterworth.output = "sos"
    mock_config.sift.butterworth.sift_threshold = 0.015

    expected_detections = (
        "20161221T004930-005030-9182", 
        np.array([13824])
    )

    # Act
    actual_detections_generator = Butterworth()._frequency_filter_sift(sample_batch)
    actual_detections = next(actual_detections_generator)


    # Assert
    assert expected_detections[0] == actual_detections[0]
    np.allclose(expected_detections[1], actual_detections[1])
