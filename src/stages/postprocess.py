import apache_beam as beam

from datetime import datetime
from typing import Dict, Any, Tuple
from types import SimpleNamespace
from matplotlib import gridspec

import librosa
import logging
import numpy as np
import os 
import time
import pandas as pd

import requests
import math 
import matplotlib.pyplot as plt
import scipy.signal


class PostprocessLabels(beam.DoFn):
    def __init__(self, config: SimpleNamespace):
        self.config = config

        self.search_output_path_template = config.search.export_template
        self.sifted_audio_path_template = config.sift.output_path_template
        self.classification_path = config.classify.classification_path


    def process(self, element: Dict[str, Any], search_output: Dict[str, Any]):
        logging.info(f"element \n{element}")
        logging.info(f"search_output \n{search_output}")
        breakpoint()

        classifications_df = pd.DataFrame([element], columns=["audio", "start", "end", "encounter_ids", "classifications"])
        classifications_df = classifications_df.explode("encounter_ids").rename(columns={"encounter_ids": "encounter_id"})
        classifications_df["encounter_id"] = classifications_df["encounter_id"].astype(str)
        
        # TODO pool classifications in postprocessing


        search_output = search_output.rename(columns={"id": "encounter_id"})
        search_output["encounter_id"] = search_output["encounter_id"].astype(str)  # TODO do in one line
        search_output = search_output[[
            # TODO refactor to confing
            "encounter_id",
            "latitude",
            "longitude",
            "displayImgUrl",
            # "species",  # TODO add in geo search stage (require rm local file)
        ]]

        # join dataframes
        joined_df = pd.merge(search_output, classifications_df, how="inner", on="encounter_id")

        logging.info(f"Final output: \n{joined_df.head()}")


        return joined_df.to_dict(orient="records")
