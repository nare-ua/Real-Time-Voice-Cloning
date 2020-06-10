docker run --gpus 1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -it --rm --expose 8888 --ipc=host \
  -e DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/snd \
  -v $PWD:/workspace/rtvc/ -v /mnt/datasets/MJ:/workspace/MJ rtvc
