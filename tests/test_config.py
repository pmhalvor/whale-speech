from config import load_pipeline_config


def test_load_pipeline_config():
    """
    There are more params present in pipeline options than only form local config.
    This test checks that all expected params are present in the pipeline options,
    via set reduction set(a) - set(b) == 0.
    """
    expected_keys = [
        'general', 
        'input', 
        'search', 
        'audio',
        'sift',
        'classify', 
        'postprocess',
        'bigquery'
    ]

    actual_config = load_pipeline_config().__dict__
    actual_keys = actual_config.keys()

    assert actual_config is not None
    assert len(expected_keys) == len(actual_keys)
    assert list(expected_keys) == list(actual_keys)
