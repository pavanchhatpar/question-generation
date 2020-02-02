FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /tf/src
COPY ./requirements.txt requirements.txt
RUN apt-get -y update\
    && apt-get install -y graphviz\
    && pip install -r requirements.txt