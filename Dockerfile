# FROM nvcr.io/nvidia/pytorch:22.10-py3
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

ARG ARTIFACTORY_CREDS_USR
ARG ARTIFACTORY_CREDS_PSW
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

