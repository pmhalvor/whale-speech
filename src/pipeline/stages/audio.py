from apache_beam.io import filesystems
from datetime import timedelta, datetime
from six.moves.urllib.request import urlopen  # pyright: ignore

import apache_beam as beam
import io
import logging
import numpy as np
import pandas as pd
import soundfile as sf

from config import load_pipeline_config


config = load_pipeline_config()


class RetrieveAudio(beam.DoFn):
    def process(self, search_results):
        """
        Takes in seach results df with encounters of whale for input date,
        and retrieves the corresponding audio. 

        If time-stamps + margin are overlapping, join the encounter ids. 
        """

        preprocessed_df = self._preprocess(search_results)

        logging.info(f"Preprocessed audio df: \n{preprocessed_df.head()}")

        # # Load the audio
        for row in preprocessed_df.itertuples():
            start_time = row.start_time
            end_time = row.end_time

            logging.info(f"Loading audio for {start_time} to {end_time}")
            audio, audio_seconds = self._load(
                start_time,
                end_time,
                config.audio.source_sample_rate,
            )

            # Yield the audio and the search_results
            yield audio, start_time, end_time, row.encounter_ids


    def _preprocess(self, df):
        df = self.__build_time_frames(df)
        df = self.__find_overlapping(df)

        return df


    def __build_time_frames(self, df):
        margin = config.audio.margin

        df["startTime"] = pd.to_datetime(df["startDate"].astype(str) + ' ' + df["startTime"].astype(str))
        df["endTime"] = pd.to_datetime(df["startDate"].astype(str) + ' ' + df["endTime"].astype(str))

        df["start_time"] = df.startTime - timedelta(seconds=margin)
        df["end_time"] = df.endTime + timedelta(seconds=margin)

        # offset TODO remove this. Only testing for now
        df["start_time"] = df["start_time"] - timedelta(hours=config.audio.offset)
        df["end_time"] = df["end_time"] - timedelta(hours=config.audio.offset)

        assert pd.api.types.is_datetime64_any_dtype(df["start_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["end_time"])

        return df
    
    
    def __find_overlapping(self, df):
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
            year,
            month,
            day,
        ):
        filename = str(config.audio.filename_template).format(year=year, month=month, day=day)
        url = config.audio.url_template.format(year=year, month=month, day=day, filename=filename)

        return url


    def _load(
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

        print(f'\n==> Loading segment from {year}-{month}-{day} @ \
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


class WriteAudio(beam.DoFn):
    def process(self, element):
        array = element[0]
        start = element[1]
        end = element[2]
        encounter_ids = element[3]

        file_path_prefix = "{date}-{start_time}-{end_time}-{ids}".format(
            date=start.strftime("%Y%m%d"),
            start_time=start.strftime("%H%M"),
            end_time=end.strftime("%H%M"),
            ids="_".join(encounter_ids)
        )

        # Create a unique file name for each element
        filename = f"{file_path_prefix}.npy"  # f"{file_path_prefix}_{hash(element)}.npy"

        file_path = config.audio.output_path_template.format(
            year=start.year,
            month=start.month,
            filename=filename
        )

        logging.info(f"Writing audio to {file_path}")
        logging.info(f"Audio shape: {array.shape}")
                
        # Write the numpy array to the file as .npy format
        with beam.io.filesystems.FileSystems.create(file_path) as f:
            np.save(f, array)  # Save the numpy array in .npy format
            
        yield file_path