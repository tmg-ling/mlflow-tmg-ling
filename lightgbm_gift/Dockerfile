#FROM python:3.8.8-slim-buster
FROM continuumio/miniconda3

# Working Directory
WORKDIR /

COPY requirements.txt ./

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000
