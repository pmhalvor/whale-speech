from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stages.sift import BaseSift, Butterworth
from types import SimpleNamespace


@pytest.fixture
def config():
    return SimpleNamespace(
        audio=SimpleNamespace(
            source_sample_rate = 16_000,
        ),
        sift=SimpleNamespace(
            output_path_template="template",
            store_sift_audio=True, 
            max_duration = 600,
            threshold = 0.015,
            window_size = 512,
            plot = True,
            plot_path_template = "plot_path_template",
            show_plots = True,
            output_array_path_template = "output_array_path_template",
            output_table_path_template = "output_table_path_template",
            project = "project",
            dataset_id = "dataset_id",
            sift_table_id = "sift_table_id",
            temp_location = "temp_location",
            sift_table_schema = SimpleNamespace(
                encounter_id=SimpleNamespace(type="STRING", mode="REQUIRED"),
            ),
            workbucket = "workbucket",
            write_params = {},
            butterworth=SimpleNamespace(
                params_path_template = "template",
                lowcut = 50,
                highcut = 1500,
                order = 2,
                output = "sos",
                sift_threshold = 0.015
            )
        ),
        general=SimpleNamespace(
            debug = True,
            filesystem = "local",
            project="project", 
            dataset_id="dataset_id",
            workbucket = "workbucket",
            temp_location="temp_location",
            show_plots = False,
        ),
        bigquery = SimpleNamespace(write_disposition="write_disposition"), 
    )

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
    audio_path = "gs://project/dataset/data/audio/raw/20240707T230800-231800.wav"
    yield  audio, start_time, end_time, encounter_ids, audio_path


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


def test_build_key(config, sample_audio_results_row):
    # Assemble
    _, start, end, encounter_ids, _ = sample_audio_results_row
    
    expected_key = "20240707T230800-231800_encounter1_encounter2"

    # Act
    actual_key = BaseSift(config)._build_key(start, end, encounter_ids)

    # Assert
    assert expected_key == actual_key


def test_preprocess(config, sample_audio_results_row):
    # Assemble
    expected_data = [
        ("20240707T230800-231800_encounter1_encounter2", np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*16_000*60)),
        ("20240707T230800-231800_encounter1_encounter2", np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*500*60)),
    ]

    # Act
    actual_data_generator = Butterworth(config)._preprocess(sample_audio_results_row)

    # Assert
    for expected in expected_data:
        actual = next(actual_data_generator)  # unload generator

        assert expected[0] == actual[0] # key
        assert expected[1].shape == actual[1].shape # data


def test_postprocess(config, sample_audio_results_row):
    # Assemble
    pcoll = sample_audio_results_row 
    min_max_detections = {
        "20240707T230800-231800_encounter1_encounter2": {"min":0, "max": 5}  # samples
    }

    expected_data = [(
        np.array([0, 1, 2, 3, 4]),          # audio
        datetime(2024, 7, 7, 23, 8, 0),     # start_time
        datetime(2024, 7, 7, 23, 18, 0),    # end_time
        ["encounter1", "encounter2"],       # encounter_ids
        'No sift audio path stored.', 
        'No detections path stored.'
    )]

    # Act
    actual_data = Butterworth(config)._postprocess(pcoll, min_max_detections)

    # Assert
    assert len(expected_data) == len(actual_data)
    for expected, actual in zip(expected_data[0], actual_data[0]):
        assert np.array_equal(expected, actual)


def test_butter_bandpass(config):
    # Assemble
    expected_coefficients = [
        [ 0.05711683,  0.11423366,  0.05711683,  1.        , -1.22806805,         0.4605427 ],
        [ 1.        , -2.        ,  1.        ,  1.        , -1.97233136,         0.97273604]
    ]

    actual_coefficients = Butterworth(config)._butter_bandpass(
        lowcut=50,
        highcut=1500,
        sample_rate=16_000,
        order=2,
        output="sos"
    )

    # Assert
    assert len(actual_coefficients) == 2  # order
    assert np.allclose(expected_coefficients, actual_coefficients)


def test_frequency_filter_sift(config, sample_batch):
    # Assemble
    expected_detections = (
        "20161221T004930-005030-9182", 
        np.array([13824])
    )

    # Act
    actual_detections_generator = Butterworth(config)._frequency_filter_sift(sample_batch)
    actual_detections = next(actual_detections_generator)


    # Assert
    assert expected_detections[0] == actual_detections[0]
    np.allclose(expected_detections[1], actual_detections[1])
