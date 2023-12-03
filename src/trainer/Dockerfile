FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://mlops_fall2023/model
ENV BUCKET_NAME=mlops_fall2023

COPY trainer /trainer

COPY modelmonitoring-406915-bda8d815d315.json /trainer


RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv google-cloud-logging google-auth



ENTRYPOINT ["python", "-m", "trainer.train"]