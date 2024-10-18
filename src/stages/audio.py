from apache_beam.io import filesystems
from datetime import timedelta, datetime
from functools import partial
from six.moves.urllib.request import urlopen  # pyright: ignore
from typing import List

import apache_beam as beam
import io
import logging
import numpy as np
import os
import pandas as pd
import soundfile as sf

from config import load_pipeline_config


config = load_pipeline_config()


class AudioTask(beam.DoFn):

    def __init__(self, config):
        self.debug = config.general.debug
        self.filesystem = config.general.filesystem.lower()

        self.margin = config.audio.margin
        self.offset = config.audio.offset
        self.source_sample_rate = config.audio.source_sample_rate
        self.url_template = config.audio.url_template

        self.filename_template = config.audio.filename_template
        self.output_array_path_template = config.audio.output_array_path_template
        self.output_table_path_template = config.audio.output_table_path_template
        self.skip_existing = config.audio.skip_existing

        self.store = config.audio.store_audio
        self.project = config.general.project
        self.dataset_id = config.general.dataset_id
        self.table_id = config.audio.audio_table_id
        self.schema = self._schema_to_dict(config.audio.audio_table_schema)
        self.workbucket = config.general.workbucket
        self.temp_location = config.general.temp_location
        self.write_params = config.bigquery.__dict__
    
    @staticmethod
    def _build_key(
            start_time: datetime,
            end_time: datetime,
            encounter_ids: list,
        ):
        start_str = start_time.strftime('%Y%m%dT%H%M%S')
        end_str = end_time.strftime('%H%M%S')
        encounter_str = "_".join(encounter_ids)
        return f"{start_str}-{end_str}_{encounter_str}"

    @staticmethod
    def _load_audio(file_path:str):
        # Write the numpy array to the file as .npy format
        with beam.io.filesystems.FileSystems.open(file_path) as f:
            audio = np.load(f)
        return audio
    
    @staticmethod
    def _schema_to_dict(schema):
        return {
            "fields": [
                {
                    "name": name, 
                    "type": getattr(schema, name).type, 
                    "mode": getattr(schema, name).mode
                } 
                for name in vars(schema)
            ]
        }

    def _get_export_path(
            self,
            key: str,
            start: datetime,
    ):
        filename = self.filename_template.format(
            year=start.year,
            month=start.month,
            day=start.day,
        ).replace("T000000Z", start.strftime("T%H%M%SZ")).replace(".wav", ".npy")

        file_path = self.output_array_path_template.format(
            key=key,
            filename=filename
        )

        if self.filesystem == "gcp":
            file_path = os.path.join(
                self.workbucket,
                file_path
            )

        return file_path

    def _file_exists_for_input(
            self, 
            key: str,
            start: datetime,
        ) -> bool:
        file_path = self._get_export_path(key, start)
        return filesystems.FileSystems.exists(file_path)
    

class RetrieveAudio(AudioTask):
    def process(self, search_results: pd.DataFrame):
        """
        Takes in seach results df with encounters of whale for input date,
        and retrieves the corresponding audio. 

        If time-stamps + margin are overlapping, join the encounter ids. 
        """
        preprocessed_df = self._preprocess(search_results)
        logging.debug(f"Preprocessed audio df: \n{preprocessed_df.head()}")

        # # Load the audio
        for row in preprocessed_df.itertuples():
            start_time = row.start_time
            end_time = row.end_time
            key = self._build_key(start_time, end_time, row.encounter_ids)

            logging.info(f"Checking if audio exists for {key}")
            if self._file_exists_for_input(key, start_time):
                audio_path = self._get_export_path(key, start_time)
                logging.info(f"Audio already exists for {key}")
                if self.skip_existing:
                    logging.info(f"Skipping downstream processing for {audio_path}")
                    continue
                else:
                    logging.info(f"Loading audio from {audio_path}")
                    audio = self._load_audio(audio_path)

            else:
                logging.info(f"Downloading audio for {key}")
                audio, _ = self._download_audio(
                    start_time,
                    end_time,
                    self.source_sample_rate,
                )

            audio_path = self._store(key, audio, start_time, end_time) if self.store else None

            yield (audio, start_time, end_time, row.encounter_ids, audio_path)
        #     preprocessed_rows.append((audio, start_time, end_time, row.encounter_ids, audio_path))

        # return preprocessed_rows

    def _preprocess(self, df: pd.DataFrame):
        df = self._build_time_frames(df)
        df = self._find_overlapping(df)

        return df

    def _build_time_frames(self, df: pd.DataFrame):
        margin = self.margin

        df["startTime"] = pd.to_datetime(df["startDate"].astype(str) + ' ' + df["startTime"].astype(str))
        df["endTime"] = pd.to_datetime(df["startDate"].astype(str) + ' ' + df["endTime"].astype(str))

        df["start_time"] = df.startTime - timedelta(seconds=margin)
        df["end_time"] = df.endTime + timedelta(seconds=margin)

        df["start_time"] = df["start_time"] - timedelta(hours=self.offset)
        df["end_time"] = df["end_time"] - timedelta(hours=self.offset)

        assert pd.api.types.is_datetime64_any_dtype(df["start_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_time"])

        return df
    
    def _find_overlapping(self, df: pd.DataFrame):
        """
        Find overlapping time-frames and join the encounter ids.  
        """
        df = df.sort_values(by="start_time").copy()
        df["encounter_id"] = df["id"].astype(str)
        
        # List to keep track of which encounters have been grouped
        visited = set()
        grouped_encounters = []

        # Overlap grouping logic
        for i, row in df.iterrows():
            min_start_time = row['start_time']
            max_end_time = row['end_time']

            if i in visited:
                continue  # Skip if this encounter is already grouped
            group = [row['encounter_id']]
            for j, other_row in df.iterrows():
                if i != j and j not in visited:
                    # Check if the rows overlap within the time margin
                    if row['start_time'] <= other_row['end_time'] and row['end_time'] >= other_row['start_time']:
                        group.append(other_row['encounter_id'])
                        visited.add(j)

                        min_start_time = min(min_start_time, other_row['start_time'])
                        max_end_time = max(max_end_time, other_row['end_time'])

            visited.add(i)
            grouped_encounters.append({
                'encounter_ids': group,
                'start_time': min_start_time,
                'end_time': max_end_time,
            })

        return pd.DataFrame.from_dict(grouped_encounters)

    def _get_file_url(
            self,
            year: int,
            month: int,
            day: int,
        ):
        filename = str(self.filename_template).format(year=year, month=month, day=day)
        url = self.url_template.format(year=year, month=month, day=day, filename=filename)

        return url

    def _download_audio(
            self,
            start_time: datetime,
            end_time: datetime,
            sample_rate: int,
        ):
        """
        Instead of downloading the whole day file, we instead only download enough data
        to cover the desired time-frame indicated by start_time and end_time.
        This means, the earlier in the day a detection was found, the less data we need to download.

        TODO: Add a check to see if the file is already downloaded and use that instead.
        TODO: Add timezone logic (needed when adding new audio sources)
        """
        year = start_time.year
        month = start_time.month
        day = start_time.day

        # starting at 00h:25m:
        at_hour = start_time.hour
        at_minute = start_time.minute

        # and with a 30-min duration:
        hours = (end_time - start_time).seconds // 3600 % 24 # leftover hours for every day
        minutes = (end_time - start_time).seconds // 60 % 60 # leftover minutes for every hour

        url = self._get_file_url(year, month, day)

        # TODO parse into a separate function
        # Note: include some space for the header of the file
        tot_audio_minutes = (at_hour + hours) * 60 + at_minute + minutes
        tot_audio_seconds = 60 * tot_audio_minutes
        tot_audio_samples = sample_rate * tot_audio_seconds

        tot_audio_bytes = 3 * tot_audio_samples    # 3 because audio is 24-bit
        max_file_size_dl = 1024 + tot_audio_bytes  # 1024 enough to cover file header

        logging.info(f'\n==> Loading segment from {year}-{month}-{day} @ \
                {at_hour}h:{at_minute}m with duration {hours}h:{minutes}m')
        psound, _ = sf.read(io.BytesIO(urlopen(url).read(max_file_size_dl)), dtype='float32')
        # (sf.read also returns the sample rate but we already know it is 16_000)

        # Get psound_segment from psound based on offset determined by at_hour:at_minute:
        offset_seconds = (at_hour * 60 + at_minute) * 60
        offset_samples = sample_rate * offset_seconds
        psound_segment = psound[offset_samples:]

        # free up RAM
        del psound

        # The size of psound_segment in seconds as desired is:
        # psound_segment_seconds = (hours * 60 + minutes) * 60
        psound_segment_seconds = psound_segment.shape[0] / sample_rate

        # TODO change prints to logging
        logging.info(f"Number of samples in segment: {psound_segment.shape[0]}") 
        logging.info(f"Numbers of seconds of segment: {psound_segment_seconds}")

        return psound_segment, psound_segment_seconds

    def _store(
        self, 
        key: str,
        audio: np.array, 
        start: datetime, 
        end: datetime, 
    ):
        file_path = self._get_export_path(key, start)
        logging.info(f"Writing audio to {file_path}")
        logging.info(f"Audio shape: {audio.shape}")
  
        if self.filesystem == "local":
            logging.info(f"Updating table {self.table_id} locally")
            self._store_local(key, file_path, start, end)
        elif self.filesystem == "gcp":
            logging.info(f"Updating table {self.table_id} in BigQuery")
            self._store_bigquery(key, file_path, start, end)
        else:
            raise ValueError(f"Invalid filesystem: {self.filesystem}")

        with filesystems.FileSystems.create(file_path) as f:
            # same for local and gcs storage
            np.save(f, audio)
        

        logging.info(f"Audio stored at {file_path}")
        return file_path

    def _store_local(
        self, 
        key:str,
        audio_path:np.array, 
        start:datetime, 
        end:datetime, 
    ):
        table_path = self.output_table_path_template.format(table_id=self.table_id)

        if not filesystems.FileSystems.exists(table_path):
            # build parent dir if necessary 
            if not filesystems.FileSystems.exists(table_path):
                filesystems.FileSystems.create(table_path)
            df = pd.DataFrame([{
                "key": key,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "audio_path": audio_path
            }])
            df.to_json(table_path, index=False, orient="records")
        else:
            df = pd.read_json(table_path, orient="records")
            new_row = pd.DataFrame([{
                "key": key,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "audio_path": audio_path
            }])
            df = pd.concat([df, new_row])
            df = df.drop_duplicates()
            df.to_json(table_path, index=False, orient="records")

    def _store_bigquery(
        self, 
        key: str,
        audio_path: str, 
        start: datetime, 
        end: datetime, 
    ):
        [{
            "key": key,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "audio_path": audio_path
        }] | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            self.table_id,
            dataset=self.dataset_id,
            project=self.project,
            schema=self.schema,
            custom_gcs_temp_location=self.temp_location,
            **self.write_params
        )
