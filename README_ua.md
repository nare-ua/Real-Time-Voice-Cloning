## build docker
`bash docker_build.sh`

## get pretrained model
from [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models)
`unzip pretrained.zip`

## run docker interactively
`bash docker_interactive.sh`

## test (inside docker)
`python demo_cli.py --no_sound`

## Run with x-window (to run demo_toolbox.py)
`bash docker_interactive_x.sh`

on the host machine

`xhost +local:$(docker inspect --format='{{ .Config.Hostname }}' ${containderId})

after getting `containerId` from `docker ps`


## demo app


### How to run

0. `git clone git@github.com:nare-ua/Real-Time-Voice-Cloning.git`
1. run `bash scripts/docker/run_demo.sh`
2. the app should be avialable at port 32774
3. Follow `security walkaround` seciton in the below 

#### samples related audio input 
https://webrtc.github.io/samples/src/content/devices/input-output/


#### chrome security walkaround
[Enabling the Microphone/Camera in Chrome for (Local) Unsecure Origins](https://medium.com/@Carmichaelize/enabling-the-microphone-camera-in-chrome-for-local-unsecure-origins-9c90c3149339)

To ignore Chrome’s secure origin policy, follow these steps.

1. Navigate to `chrome://flags/#unsafely-treat-insecure-origin-as-secure` in Chrome.
2. Find and enable the `Insecure origins treated as secure` section (see below).
3. Add any addresses you want to ignore the secure origin policy for. Remember to include the port number too (if required).
3. Save and restart Chrome.

