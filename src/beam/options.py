from apache_beam.options.pipeline_options import PipelineOptions

import apache_beam as beam
import os 
import yaml

CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'beam_config.yaml'
)



class GeneralOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--verbose', type=str, help='Verbose logging', required=False)


class InputOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--start', type=str, help='Datetime str for start of search', required=False)
        parser.add_value_provider_argument('--end', type=str, help='Datetime str for end of search', required=False)


class SearchOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--search_filename', type=str, help='Geometry search filename', required=False)
        parser.add_value_provider_argument('--species', type=str, help='Species to search for (default humpback_whale)', required=False)


class AudioOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--filename_template', type=str, help='Filename template for audio files (ex. {year}{month}{day}_{hour}{minute}_{species}.wav)', required=False)
        parser.add_value_provider_argument('--path_template', type=str, help='Path template for audio files (ex. {workbucket}/{filename})', required=False)
        parser.add_value_provider_argument('--sample_rate', type=int, help='Sample rate for audio files', required=False)
        parser.add_value_provider_argument('--margin', type=int, help='Margin for audio files', required=False)


class DetectionFilterOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--highcut', type=int, help='Highcut frequency for audio filtering', required=False)
        parser.add_value_provider_argument('--lowcut', type=int, help='Lowcut frequency for audio filtering', required=False)
        parser.add_value_provider_argument('--order', type=int, help='Order for audio filtering', required=False)
        parser.add_value_provider_argument('--frequency_threshold', type=float, help='Threshold for audio filtering', required=False)
        parser.add_value_provider_argument('--window_size', type=int, help='Window size for audio filtering', required=False)


class ModelOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--url', type=str, help='URL for model', required=False)
        parser.add_value_provider_argument('--model_sample_rate', type=int, help='Sample rate for model', required=False)
        parser.add_value_provider_argument('--hydrophone_sensitivity', type=float, help='Hydrophone sensitivity for model', required=False)


class PostprocessOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_value_provider_argument('--min_gap', type=int, help='Minimum gap for postprocessing', required=False)
        parser.add_value_provider_argument('--confidence_threshold', type=float, help='Threshold for postprocessing')
        parser.add_value_provider_argument('--pooling', type=str, help='Pooling mechanism for confidence scores')
        parser.add_value_provider_argument('--output_path_template', type=str, help='Output path template for final results')


# Function to load defaults from config file
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_pipeline_options(config_file = CONFIG_FILE_PATH):
    # Load pipeline options from the config file
    config = load_config(config_file).get('pipeline')

    # Initialize pipeline options for each section
    pipeline_options = PipelineOptions()

    # general_options = pipeline_options.view_as(GeneralOptions).from_dictionary(config['general']) # skip boolean for now
    input_options = pipeline_options.view_as(InputOptions).from_dictionary(config['input'])
    search_options = pipeline_options.view_as(SearchOptions).from_dictionary(config['search'])
    audio_options = pipeline_options.view_as(AudioOptions).from_dictionary(config['audio'])
    detection_filter_options = pipeline_options.view_as(DetectionFilterOptions).from_dictionary(config['detection_filter'])
    model_options = pipeline_options.view_as(ModelOptions).from_dictionary(config['model'])
    postprocess_options = pipeline_options.view_as(PostprocessOptions).from_dictionary(config['postprocess'])
        
    # Set other global pipeline options
    pipeline_options.view_as(beam.options.pipeline_options.SetupOptions).save_main_session = True
    
    return pipeline_options


if __name__ == "__main__":
    print(load_pipeline_options(CONFIG_FILE_PATH).display_data().keys())
