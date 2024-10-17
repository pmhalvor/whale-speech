FROM apache/beam_python3.11_sdk

COPY . /home

WORKDIR /home

# Install HDF5 using apt
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install -r requirements.txt

CMD ["python", "src/pipeline/app.py"]
