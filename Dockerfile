FROM apache/beam_python3.11_sdk

COPY src/ /home/src/
COPY requirements/requirements.txt /home/requirements.txt
COPY data/geo /home/data/geo

WORKDIR /home

# Install HDF5 using apt
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libsndfile1 \
    gcc
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "src/pipeline.py"]
