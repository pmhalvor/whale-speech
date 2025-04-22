gcloud run deploy  model-server-local \
--add-volume name=whale-speech-data,type=cloud-storage,bucket=gs://bioacoustic/whale-speech/data \
--add-volume-mount volume=whale-speech-data,mount-path=/home/data \
--image us-central1-docker.pkg.dev/bioacoustics-2024/whale-speech/model-server@sha256:711525c13d80aafa6598297d48e48e1bd5f90d468a54aafc07b77a087f31bb75 \
--region us-central1 \
--port='5000'