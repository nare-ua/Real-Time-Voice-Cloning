#!/bin/bash

set -euxo pipefail

dataroot="/mnt/data"

docker run --gpus 0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 32774:10050 --ipc=host -v $PWD:/workspace/rtvc/ -v ${dataroot}:/mnt/data rtvc /opt/conda/bin/python /workspace/rtvc/app/app.py
