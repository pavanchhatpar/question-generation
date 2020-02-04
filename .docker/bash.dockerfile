FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /tf/src
RUN apt-get -y update\
    && apt-get install -y graphviz
COPY . /tf/src
RUN pip install -r requirements.txt