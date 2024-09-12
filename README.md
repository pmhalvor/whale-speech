# 📣🐋 Whale Speech 
A pipeline to map whale sightings to hydrophone audio.
## Installation
```bash
conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```

## Pipeline description

Stages:
1. **Input**: When (and where*) to look for whale sightings on [HappyWhale](https://happywhale.com/).
2. **Geometry Search**: Query the [open-oceans/happywhale](https://github.com/open-oceans/happywhale) to find potential whale sightings. 
3. **Retrive Audio**: Download audio from MBARI's [Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/) around the time of the sighting. 
4. **Filter Frequency**: Break audio into non-overlaping segments with flagged frequency detections. 
5. **Classify Audio**: Use a NOAA and Google's [humpback_whale model](https://tfhub.dev/google/humpback_whale/1) to classify the audio segments.
6. **Postprocess Labels**: Build clip-intervals for each sighting for playback snippets.
7. **Output**: Map the whale sightings to the audio segments.


[![](https://mermaid.ink/img/pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-?type=png)](https://mermaid.live/edit#pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-)





*Currently only support sightings around the Monterey Bay Hydrophone ([MARS](https://www.mbari.org/technology/monterey-accelerated-research-system-mars/)).

## Resources 
- [HappyWhale](https://happywhale.com/)
- [open-oceans/happywhale](https://github.com/open-oceans/happywhale)
- [MBARI's Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/)
- [NOAA and Google's humpback_whale model](https://tfhub.dev/google/humpback_whale/1)
- [Monterey Bay Hydrophone](https://www.mbari.org/technology/monterey-accelerated-research-system-mars/)
