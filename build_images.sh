
docker build -t cs8674_$1 -f ./$1/.docker/bash.dockerfile ./$1
docker build -t cs8674_$1_nb -f ./$1/.docker/nb.dockerfile ./$1