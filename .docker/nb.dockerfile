FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
WORKDIR /tf/src
RUN apt-get -y update\
    && apt-get install -y graphviz
COPY . /tf/src
RUN pip install -r requirements.txt\
    && jupyter notebook --generate-config\
    && mv /root/.jupyter /.jupyter\
    && v="from notebook.auth import passwd; print(passwd('`cat jupyter.password`'))"\
    && python -c "$v" > tmp.txt\
    && echo "c.NotebookApp.password='`cat tmp.txt`'" >> /.jupyter/jupyter_notebook_config.py