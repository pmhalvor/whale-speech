import pytest
import pandas as pd

from datetime import datetime
from types import SimpleNamespace
from stages.postprocess import PostprocessLabels

@pytest.fixture
def config():
    return SimpleNamespace(
        search=SimpleNamespace(output_path_template="template"),
        sift=SimpleNamespace(output_array_path_template="template"),
        classify=SimpleNamespace(output_path_template="path"),
        postprocess=SimpleNamespace(
            pooling="mean", 
            postprocess_table_id="table_id",
            output_path="output_path",
            postprocess_table_schema=SimpleNamespace(
                start=SimpleNamespace(type="TIMESTAMP", mode="REQUIRED"),
                end=SimpleNamespace(type="TIMESTAMP", mode="REQUIRED"),
                encounter_id=SimpleNamespace(type="STRING", mode="REQUIRED"),
                pooled_score=SimpleNamespace(type="FLOAT", mode="REQUIRED"),
            ),
        ),
        general=SimpleNamespace(project="project", dataset_id="dataset_id", filesystem="local"),
    )

@pytest.fixture
def element():
    return {
        "audio": "audio",
        "start": datetime(2024, 9, 10, 11, 12, 13),
        "end": datetime(2024, 9, 10, 12, 13, 14),
        "encounter_ids": ["a123", "b456"],
        "classifications": [1, 2, 3],
        "classification_path": "gs://project/dataset/data/classifications/file"
    }

@pytest.fixture
def search_output():
    return pd.DataFrame({
        "id": ["a123", "b456", "c789"],
        "latitude": [1.0, 2.0, 3.0],
        "longitude": [1.1, 2.2, 3.3],
        "displayImgUrl": ["example.com/a123", "example.com/b456", "example.com/c789"],
        "extra_column": ["extra1", "extra2", "extra3"]
    })


def test_build_classification_df(config, element):
    # Arrange
    postprocess_labels = PostprocessLabels(config)

    expected = pd.DataFrame([
        {
            "start": "2024-09-10T11:12:13",
            "end": "2024-09-10T12:13:14",
            "encounter_id": "a123",
            "classification_path": "gs://project/dataset/data/classifications/file",
            "pooled_score": 2.0,
        },
        {
            "start": "2024-09-10T11:12:13",
            "end": "2024-09-10T12:13:14",
            "encounter_id": "b456",
            "classification_path": "gs://project/dataset/data/classifications/file",
            "pooled_score": 2.0,
        }
    ])

    # Act
    actual = postprocess_labels._build_classification_df(element)

    # Assert
    for e, a in zip(expected, actual):
        assert e == a


def test_build_search_output_df(config, search_output):
    # Arrange
    postprocess_labels = PostprocessLabels(config)

    expected = pd.DataFrame({
        "encounter_id": ["a123", "b456", "c789"],
        "latitude": [1.0, 2.0, 3.0],
        "longitude": [1.1, 2.2, 3.3],
        "displayImgUrl": ["example.com/a123", "example.com/b456", "example.com/c789"],
    })

    # Act
    actual = postprocess_labels._build_search_output_df(search_output)

    # Assert
    assert expected.equals(actual)


def test_pool_classifications(config):
    # Arrange
    postprocess_labels = PostprocessLabels(config)
    classifications = [1, 2, 3, 4]

    # Act
    actual = postprocess_labels._pool_classifications(classifications)

    # Assert
    assert actual == 2.5  # note only checks mean, update for more


def test_add_paths(config, search_output):
    # Arrange
    postprocess_labels = PostprocessLabels(config)

    expected = pd.DataFrame({
        # reusing same data as above for simplicity
        "id": ["a123", "b456", "c789"],
        "latitude": [1.0, 2.0, 3.0],
        "longitude": [1.1, 2.2, 3.3],
        "extra_column": ["extra1", "extra2", "extra3"],
        
        # added path columns
        "img_path": ["example.com/a123", "example.com/b456", "example.com/c789"],
        "audio_path": ["NotImplemented", "NotImplemented", "NotImplemented"],
        "classification_path": ["NotImplemented", "NotImplemented", "NotImplemented"],
    })

    # Act
    actual = postprocess_labels._add_paths(search_output, {})

    # Assert
    print(expected.columns)
    print(actual.columns)
    for e, a in zip(expected, actual):
        assert e == a
