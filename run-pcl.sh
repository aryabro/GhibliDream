#!/bin/bash
# FOR WSL, comment on linux
export LIBGL_ALWAYS_INDIRECT=1
export DISPLAY=$(ip route|awk '/^default/{print $3}'):0.0

sudo docker run \
    -it \
    --rm \
    --gpus='all,"capabilities=compute,utility,graphics"' \
    --network host \
    --privileged \
    -v .:/work \
    --workdir /work \
    --env-file env.list \
    --name $1 \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    humble-img:v2 \
    /bin/bash

    # -v $PWD:/work \
