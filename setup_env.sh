if [ ! -f ".env" ]; then
    echo "Refer to sample.env to make .env file as per your project"
    echo "See README for more details"
    exit 2
fi

if [ -f "lock.env" ]; then
    source "lock.env"
    docker rmi cs8674/nb:latest
    docker rmi cs8674/bash:latest
    rm -rf $PYTHON_VENV_PATH
    rm ./data
    rm "lock.env"
fi

source ".env"
if [ -z "$DATA_DIR" ]; then
    echo ".env should set DATA_DIR"
    exit 1
fi
if [ -z "$PYTHON_VENV_PATH" ]; then
    echo ".env should set PYTHON_VENV_PATH"
    exit 1
fi
if [ -z "$JUPYTER_PASSWD" ]; then
    echo ".env should set JUPYTER_PASSWD"
    exit 1
fi

ln -s $DATA_DIR ./data

docker build --build-arg JUPYTER_PASSWD --target bash -t cs8674/bash:latest .
docker build --build-arg JUPYTER_PASSWD --target nb -t cs8674/nb:latest .

python3 -m venv $PYTHON_VENV_PATH
source $PYTHON_VENV_PATH/bin/activate\
    && pip install --upgrade pip setuptools\
    && pip install --no-cache-dir -r requirements.txt

cp ".env" "lock.env"

echo "***Environment is ready***"
echo ""
echo "-- Docker images prepared --"
echo "cs8674/bash:latest"
echo "cs8674/nb:latest"
echo ""
echo "-- Run commands --"
echo "./run.sh bash"
echo "./run.sh nb"