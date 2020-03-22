if [ ! -f "lock.env" ]; then
    echo "run ./setup_env.sh first"
    exit 1
fi

source "lock.env"

if [ "$1" == "nb" ]; then
    docker run -it --gpus all --rm\
        -u $(id -u):$(id -g)\
        -v "`pwd`:/tf/src"\
        -v "`echo $DATA_DIR`:/tf/data"\
        -p 8991:8888\
        cs8674/$1:latest
else
    docker run -it --gpus all --rm\
        -u $(id -u):$(id -g)\
        -v "`pwd`:/tf/src"\
        -v "`echo $DATA_DIR`:/tf/data"\
        cs8674/$1:latest
fi