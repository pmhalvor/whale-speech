# 📣🐋 Whale Speech 
A pipeline to map whale sightings to hydrophone audio.
## Installation
M1:
```bash
CONDA_SUBDIR=osx-arm64 conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```


Other:
```bash
conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```

## Pipeline description

Stages:
1. **Input**: When (and where*) to look for whale sightings on [HappyWhale](https://happywhale.com/).
2. **Geometry Search**: Query [open-oceans/happywhale](https://github.com/open-oceans/happywhale) to find potential whale sightings. 
3. **Retrive Audio**: Download audio from MBARI's [Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/) around the time of the sighting. 
4. **Filter Frequency**: Break audio into non-overlaping segments with flagged frequency detections. 
5. **Classify Audio**: Use a NOAA and Google's [humpback_whale model](https://tfhub.dev/google/humpback_whale/1) to classify the flagged segments.
6. **Postprocess Labels**: Build clip-intervals for each sighting for playback snippets.
7. **Output**: Map the whale sighting ids to the playback snippets.


<!-- Light mode -->
[![](https://mermaid.ink/img/pako:eNpVkctOwzAQRX_FmhVIadWm7yyQ-i4SCERZ0XThOpPEkhMXP4BQ9d9xE4PobOw7Ptcz9pyAyQQhglTIT5ZTZcjrIi6Ji-kupVFKW4wKLBOqyH15tEbvSat1R2Y3zWEm5AHJGmWBRlVki1Sx_NZfUJNzTxZWc0ZeHMY_kExtwqXnZg3XiHktlt6UcmFQkZVfFL5bLFnljcuaXXn2YIXQWCGZC6o1T6urIquaXXvW0Iw8S22OSjLUmjzQAwrt0XWNbna_LeicPFnj3r7_3--GXBQEUKAqKE_cH54umRhMjgXGELltgim1wsQQl2eHUmvktioZREZZDEBJm-XgqrjGA7DHhBpccJopWvxlj7R8k_JKQ3SCL4h6nbDdHYfd8XAwmfTD3rAXQAVR5xzAd-3otCdNjMJBOBj1x6MAMOFGqsdm6PXszz-lFp4w?type=png)](https://mermaid.live/edit#pako:eNpVkctOwzAQRX_FmhVIadWm7yyQ-i4SCERZ0XThOpPEkhMXP4BQ9d9xE4PobOw7Ptcz9pyAyQQhglTIT5ZTZcjrIi6Ji-kupVFKW4wKLBOqyH15tEbvSat1R2Y3zWEm5AHJGmWBRlVki1Sx_NZfUJNzTxZWc0ZeHMY_kExtwqXnZg3XiHktlt6UcmFQkZVfFL5bLFnljcuaXXn2YIXQWCGZC6o1T6urIquaXXvW0Iw8S22OSjLUmjzQAwrt0XWNbna_LeicPFnj3r7_3--GXBQEUKAqKE_cH54umRhMjgXGELltgim1wsQQl2eHUmvktioZREZZDEBJm-XgqrjGA7DHhBpccJopWvxlj7R8k_JKQ3SCL4h6nbDdHYfd8XAwmfTD3rAXQAVR5xzAd-3otCdNjMJBOBj1x6MAMOFGqsdm6PXszz-lFp4w)

<!-- Dark mode -->
<!-- [![](https://mermaid.ink/img/pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-?type=png)](https://mermaid.live/edit#pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-) -->




<sub>
*Currently only support sightings around the Monterey Bay Hydrophone (<a href="https://www.mbari.org/technology/monterey-accelerated-research-system-mars/">MARS</a>).
</sub>

## Resources 
- [HappyWhale](https://happywhale.com/)
- [open-oceans/happywhale](https://github.com/open-oceans/happywhale)
- [MBARI's Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/)
- [NOAA and Google's humpback_whale model](https://tfhub.dev/google/humpback_whale/1)
- [Monterey Bay Hydrophone MARS](https://www.mbari.org/technology/monterey-accelerated-research-system-mars/)
