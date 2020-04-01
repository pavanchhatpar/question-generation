FROM tensorflow/tensorflow:latest-gpu-py3-jupyter AS nb
WORKDIR /tf/src
RUN apt-get -y update\
    && apt-get install -y graphviz
COPY . /tf/src
ARG JUPYTER_PASSWD
RUN pip uninstall -y enum34\
    && pip install -r requirements.txt\
    && jupyter notebook --generate-config\
    && mv /root/.jupyter /.jupyter\
    && v="from notebook.auth import passwd; print(passwd('`echo $JUPYTER_PASSWD`'))"\
    && python -c "$v" > tmp.txt\
    && echo "c.NotebookApp.password='`cat tmp.txt`'" >> /.jupyter/jupyter_notebook_config.py

FROM nb AS bash
CMD ["bash"]