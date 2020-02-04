docker run -it --gpus all --rm\
    -u $(id -u):$(id -g)\
    -v "`pwd`:/tf/src"\
    -v "/mnt/pavan-ssd/cs8674:/tf/data"\
    -p 8991:8888\
    cs8674_$1