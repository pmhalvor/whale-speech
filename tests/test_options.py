from beam.options import load_pipeline_options

def test_load_pipeline_options():
    """
    There are more params present in pipeline options than only form local config.
    This test checks that all expected params are present in the pipeline options,
    via set reduction set(a) - set(b) == 0.
    """
    expected_keys = [
        'save_main_session', 
        'verbose', 
        'start', 
        'end', 
        'search_filename', 
        'species', 
        'filename_template', 
        'path_template', 
        'sample_rate', 
        'margin', 
        'highcut', 
        'lowcut',
        'order', 
        'frequency_threshold', 
        'window_size', 
        'url', 
        'model_sample_rate', 
        'hydrophone_sensitivity', 
        'min_gap', 
        'confidence_threshold', 
        'pooling', 
        'output_path_template'
    ]

    options = load_pipeline_options()
    actual_keys = options.get_all_options().keys()

    assert options is not None
    assert set(expected_keys) - set(actual_keys) == set()


