## build docker
`sh docker_build.sh`

## get pretrained model
from [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models)
`unzip pretrained.zip`

## run docker interactively
`sh docker_interactive.sh`

## test (inside docker)
`python demo_cli.py`

## Run with x-window (to run demo_toolbox.py)
`sh docker_interactive_x.sh`

on the host machine

`xhost +local:$(docker inspect --format='{{ .Config.Hostname }}' ${containderId})

after getting `containerId` from `docker ps`
