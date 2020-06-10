#!/bin/bash

set -euxo pipefail

dataroot="/mnt/data"

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -P --ipc=host -v $PWD:/workspace/rtvc/ -v ${dataroot}:/mnt/data rtvc bash
