from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stages.classify import BaseClassifier, WhaleClassifier, InferenceClient
from unittest.mock import patch


@pytest.fixture
def example_config():
    return SimpleNamespace(
        general = SimpleNamespace(
            show_plots=True,
        ),
        audio = SimpleNamespace(source_sample_rate=16_000),
        classify = SimpleNamespace(
            batch_duration=30, # seconds
            hydrophone_sensitivity=-168.8,
            model_sample_rate=10_000,
            model_url="http://127.0.0.1:5000/predict",
            plot_scores=True,
            plot_path_template="data/plots/results/{year}/{month:02}/{plot_name}.png",
            med_filter_size=3,
        ),
    )

@pytest.fixture
def example_input_row_small():
    audio = np.array([0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]) # bigger than max_duration
    start_time = datetime(2024, 7, 7, 23, 8, 0)
    end_time = datetime(2024, 7, 7, 23, 18, 0)
    encounter_ids = ["encounter1", "encounter2"]
    yield  audio, start_time, end_time, encounter_ids

@pytest.fixture
def example_input_row_large():
    audio = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]*16_500*60).astype(np.float32) # bigger than max_duration
    start_time = datetime(2024, 7, 7, 23, 8, 0)
    end_time = datetime(2024, 7, 7, 23, 18, 0)
    encounter_ids = ["encounter1", "encounter2"]
    yield  audio, start_time, end_time, encounter_ids


@pytest.fixture
def example_grouped_outputs():
    return {
        "20240707T230800-231800_encounter1_encounter2": [0.3, 0.7, 0.2, 0.6]
    }


def test_preprocess_small(example_config, example_input_row_small):
    # Arrange
    input_row = example_input_row_small

    expected = (
        "20240707T230800-231800_encounter1_encounter2",
        # the resampled array 
        np.array([[0.12715314],[1.56746149],[3.21961117],[4.73285103],[3.70353985],[1.95126152],[0.]])
    )

    # Act
    actual_generator = BaseClassifier(example_config)._preprocess(input_row)
    actual = next(actual_generator)

    # Assert
    assert expected[0] == actual[0]
    assert expected[1].shape == actual[1].shape
    np.testing.assert_almost_equal(expected[1], actual[1], decimal=6)


def test_preprocess_large(example_config, example_input_row_large):
    """
    Instead of writing out the expected array, just check for length and shape
    """

    # Arrange
    input_row = example_input_row_large
    batch_duration      = example_config.classify.batch_duration
    model_sample_rate   = example_config.classify.model_sample_rate

    expected = (
        "20240707T230800-231800_encounter1_encounter2",
        # large inputs yield fixed sized batches
        np.zeros((batch_duration*model_sample_rate, 1))
    )

    # Act
    actual_generator = BaseClassifier(example_config)._preprocess(input_row)
    actual = next(actual_generator)

    # Assert
    assert expected[0] == actual[0]
    assert expected[1].shape == actual[1].shape


def test_postprocess(example_config, example_input_row_small, example_grouped_outputs):
    # Arrange
    input_row = example_input_row_small

    expected = (
        np.array([0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]), # audio
        datetime(2024, 7, 7, 23, 8, 0),                     # start_time
        datetime(2024, 7, 7, 23, 18, 0),                    # end_time
        ["encounter1", "encounter2"],                       # encounter_ids
        [0.3, 0.7, 0.2, 0.6]                                # scores
    )

    # Act
    actual = WhaleClassifier(example_config)._postprocess(input_row, example_grouped_outputs)

    # Assert
    assert len(expected) == len(actual)
    for e, a in zip(expected, actual):
        if isinstance(e, np.ndarray):
            np.testing.assert_almost_equal(e, a, decimal=6)
        else:
            assert e == a
    

def test_resample(example_config):
    # Arrange
    audio = np.array([0., 1., 2., 3., 4., 5., 4., 3., 2., 1.])
    example_config.audio.source_sample_rate = 16_000
    example_config.classify.model_sample_rate = 10_000

    expected = np.array([0.127153, 1.567461, 3.219611, 4.732851, 3.70354 , 1.951262, 0.])

    # Act
    actual = BaseClassifier(example_config)._resample(audio)

    # Assert
    np.testing.assert_almost_equal(expected, actual, decimal=6)


def test_resample_same_rate(example_config):
    # Arrange
    audio = np.array([0., 1., 2., 3., 4., 5., 4., 3., 2., 1.])
    example_config.audio.source_sample_rate = 10_000
    example_config.classify.model_sample_rate = 10_000

    expected = np.array([0., 1., 2., 3., 4., 5., 4., 3., 2., 1.])

    # Act
    actual = BaseClassifier(example_config)._resample(audio)

    # Assert
    np.testing.assert_almost_equal(expected, actual, decimal=6)


@patch('stages.classify.requests.post')
def test_process(mock_post, example_config):
    # Arrange
    mock_response = SimpleNamespace(
        json=lambda: {"key": "cool_looking_key", "predictions": [0.3, 0.7, 0.2, 0.6]},
        status_code=200,
        raise_for_status=lambda: None
    )
    mock_post.return_value = mock_response

    input_element = (
        "cool_looking_key",  # key 
        np.array([[0.12715314],[1.56746149],[3.21961117],[4.73285103],[3.70353985],[1.95126152],[0.]])  # batch
    )

    expected = (
        "cool_looking_key",             # key
        np.array([0.3, 0.7, 0.2, 0.6])  # scores
    )

    # Act
    actual_genreator = InferenceClient(example_config).process(input_element)
    actual = next(actual_genreator) 

    # Assert
    assert expected[0] == actual[0]
    np.testing.assert_almost_equal(expected[1], actual[1], decimal=6)
    