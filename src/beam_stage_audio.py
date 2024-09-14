import apache_beam as beam
from datetime import timedelta, datetime
from six.moves.urllib.request import urlopen  # pyright: ignore

import io
import soundfile as sf

import config 
stages = config.load()

## Params (unpacked here for easier updates later)
### input
start_time = datetime.strptime(stages.input.start, '%Y-%m-%dT%H:%M:%S')
end_time = datetime.strptime(stages.input.end, '%Y-%m-%dT%H:%M:%S')

### audio 
sample_rate = stages.audio.sample_rate
path_template = stages.audio.path_template
filename_template = stages.audio.filename_template  
margin = stages.audio.margin

### general 
verbose = stages.general.verbose


class RetrieveAudio(beam.PTransform):
    def expand(self, pcoll):
        return (
            pcoll
            | "Retrieve Audio" >> beam.Map(lambda x: self._load(x['result_file'], x['start_time'], x['end_time']))
        )

    def _get_file_url(
            self,
            year,
            month,
            day,
        ):
        # The url to download for that day is:
        print(f'\n==> Getting file URL for {year}-{month}-{day}') if verbose else None


        filename = str(filename_template).format(year=year, month=month, day=day)
        url = path_template.format(year=year, month=month, day=day, filename=filename)

        return url



    def _load(
            self,
            start_time,
            end_time,
            sample_rate,
        ):
        """
        Instead of downloading the whole day file, we instead only download enough data
        to cover the desired time-frame indicated by start_time and end_time.
        This means, the earlier in the day a detection was found, the less data we need to download.
        """
        if margin:
            start_time -= timedelta(seconds=margin)
            end_time += timedelta(seconds=margin)

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
        print("Number of samples in segment: ", psound_segment.shape[0]) if verbose else None
        print("Numbers of seconds of segment:", psound_segment_seconds) if verbose else None

        return psound_segment, psound_segment_seconds
