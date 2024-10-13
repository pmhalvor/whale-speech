# 📣🐋 Whale Speech 
A pipeline to map whale encounters to hydrophone audio.

<sub>
Derived from <a href="https://docs.mbari.org/pacific-sound/notebooks/humpbackwhales/detect/PacificSoundDetectHumpbackSong/"> PacificSoundDetectHumpbackSong</a>, though not directly affiliated with MBARI, NOAA, or HappyWhale.
</sub>


## Getting started

### Install
#### M1:
```bash
CONDA_SUBDIR=osx-arm64 conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```


#### Other:
```bash
conda create -n whale-speech python=3.11
conda activate whale-speech
pip install -r requirements.txt
```

### Run locally 
```bash
make local-run
```

## Pipeline description

Stages:
1. **Input**: When (and where*) to look for whale encounters on [HappyWhale](https://happywhale.com/).
2. **Geometry Search**: Query [open-oceans/happywhale](https://github.com/open-oceans/happywhale) to find potential whale encounters. 

   &rarr; Expected outputs: encounter ids, start and end times, and longitude and latitude.

3. **Retrive Audio**: Download audio from MBARI's [Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/) around the time of the encounter. 
    
    &rarr; Expected outputs: audio array, start and end times, and encounter ids.
    
4. **Filter Frequency**: Break audio into non-overlaping segments with flagged frequency detections. 
        
    &rarr; Expected outputs: cut audio array, detection intervals, and encounter ids.

5. **Classify Audio**: Use a NOAA and Google's [humpback_whale model](https://tfhub.dev/google/humpback_whale/1) to classify the flagged segments.

    &rarr; Expected outputs: resampled audio, classification score array, and encounter ids.

6. **Postprocess Labels**: Build clip-intervals for each encounter for playback snippets.

    &rarr; Expected outputs: encounter ids, cut/resampled audio array, and aggregated classification score.

7. **Output**: Map the whale encounter ids to the playback snippets.

<!-- Light mode -->
[![](https://mermaid.ink/img/pako:eNpVkl1PwjAUhv_KybnSZBAYIrAYE0VBE41GvNJxUbqzrUm3Yj_USfjvlq0aPTfte_qcz3SHXGWECeZSffCSaQvPV2kN3i5ec5bkrMeZpDpjGm7rrbNmDb3eOVwedY-FVBuCJamKrG5gRUzz8jgkaMl5ICtnBIcnj4l3gguXCRW4y47rxLwV1yEoF9KShkU4NL05qnkTAq9bdhHYjZPSUEMwl8wYkTf_iizgrN_39G2gtdooC_d-eOm71u-kf8FD0mXALCvgURm71YqTMXDHNiRNQJctevP606sp4cFZv6T138Fu4KAwwop0xUTml707eFK0JVWUYuKvGeXMSZtiWu89ypxVq6bmmFjtKEKtXFGir-InjNBtM2bpSrBCs-rXu2X1i1L_NCY7_MRkNIj7w2k8nJ6OZ7OTeHQ6irDBZLCP8KuNGPRnnU3icTyenEwnEVImrNL33e_gqs5Fgftv_9OrOw?type=png)](https://mermaid.live/edit#pako:eNpVkl1PwjAUhv_KybnSZBAYIrAYE0VBE41GvNJxUbqzrUm3Yj_USfjvlq0aPTfte_qcz3SHXGWECeZSffCSaQvPV2kN3i5ec5bkrMeZpDpjGm7rrbNmDb3eOVwedY-FVBuCJamKrG5gRUzz8jgkaMl5ICtnBIcnj4l3gguXCRW4y47rxLwV1yEoF9KShkU4NL05qnkTAq9bdhHYjZPSUEMwl8wYkTf_iizgrN_39G2gtdooC_d-eOm71u-kf8FD0mXALCvgURm71YqTMXDHNiRNQJctevP606sp4cFZv6T138Fu4KAwwop0xUTml707eFK0JVWUYuKvGeXMSZtiWu89ypxVq6bmmFjtKEKtXFGir-InjNBtM2bpSrBCs-rXu2X1i1L_NCY7_MRkNIj7w2k8nJ6OZ7OTeHQ6irDBZLCP8KuNGPRnnU3icTyenEwnEVImrNL33e_gqs5Fgftv_9OrOw)

<!-- Dark mode -->
<!-- [![](https://mermaid.ink/img/pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-?type=png)](https://mermaid.live/edit#pako:eNpVkNtOwzAMhl8l8lWRumnrzrlA2qkbEhKIcQXdRUjdNaJNSg5AmfbuZF2FmK_8-_ud2D4CVykChaxQXzxn2pLnVSKJj_lrxmjGOpwVKFOmyZ2snDV70unckkWwQVWi1TXZIdM8v2mbGroMnjwSn0jmLhWqZYsLu4hlI9ZBLAqLmsQaPxxKXrfedYPjYFkwY0RWXz0UN3ATPCpjK604GkPu2RsWpjVsGsO2XSATJicPzvrp9_8n2Z4FhFCiLplI_RGO50oCNscSE6A-9Xu_J5DIk_cxZ9Wulhyo1Q5D0ModcvA_FMYrV6XM4kqwg2blX7Vi8kWpKw30CN9AB72o259G_el4NJsNo8F4EEINtHcK4afp6HVnl5hEo2g0GU4np1-Ax4B-) -->




<sub>
*Currently only support encounters around the Monterey Bay Hydrophone (<a href="https://www.mbari.org/technology/monterey-accelerated-research-system-mars/">MARS</a>).
</sub>

## Resources 
- [HappyWhale](https://happywhale.com/)
- [open-oceans/happywhale](https://github.com/open-oceans/happywhale)
- [MBARI's Pacific Ocean Sound Recordings](https://registry.opendata.aws/pacific-sound/)
- [NOAA and Google's humpback_whale model](https://tfhub.dev/google/humpback_whale/1)
- [Monterey Bay Hydrophone MARS](https://www.mbari.org/technology/monterey-accelerated-research-system-mars/)
