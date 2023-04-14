#!/usr/bin/env bash
set -e

docker_image_name=$1
src_folder=$2

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <docker_image_name> <src_folder>"
    exit 1
fi

# Check if the source folder exists
if [ ! -d "$src_folder" ]; then
    echo "Error: Source folder does not exist."
    exit 1
fi

pub_key_file=$(find ~/.ssh -type f -name "*.pub" | head -n 1)

if [ -z "$pub_key_file" ]; then
  echo "No public key file found in ~/.ssh directory - ssh will not work"
fi

#docker run --name $docker_image_name -d -it \
#  --gpus all \
#  -p 8080:8080 \
#  -p 5900:5900 \
#  -p 8894:8894 \
#  -p 0.0.0.0:8265:8265 \
#  -p 0.0.0.0:6006:6006 \
#  -e AUTHORIZED_KEYS="`cat $pub_key_file`" \
#  -v $src_folder:/src \
#  --ipc=host \
#  -v ~/.${docker_image_name}_home:/root \
#  $docker_image_name bash >/dev/null

# docker stop ${docker_image_name}
# docker rm ${docker_image_name}
GPUS="device=4"
# GPUS="all"

docker run --name $docker_image_name -d -it \
  --gpus="\"$GPUS\",\"capabilities=compute,utility,graphics,display\"" \
  -p 8080:8080 \
  -p 5900:5900 \
  -p 8894:8894 \
  -p 0.0.0.0:8265:8265 \
  -p 0.0.0.0:6006:6006 \
  -e AUTHORIZED_KEYS="`cat $pub_key_file`" \
  -v $src_folder:/src \
  --ipc=host \
  -v ~/.${docker_image_name}_home:/root \
  ${docker_image_name} bash

# wait a bit and check if container is up
sleep 1
container_id=$(docker ps --filter "ancestor=$docker_image_name" --format "{{.ID}}")
if [ -z "$container_id" ]; then
    echo "Container failed to start!"
    docker logs ${docker_image_name}
    exit 1
fi

SANDBOX_IP="$(docker inspect -f '{{ .NetworkSettings.IPAddress }}' $docker_image_name)"

# after a rebuild, we should remove the ssh identity
ssh-keygen -f "$HOME/.ssh/known_hosts" -R $SANDBOX_IP

echo "Successfully started the sandbox!"
echo "SSH with 'ssh root@$SANDBOX_IP'"
echo "SSH with 'ssh carla@$SANDBOX_IP'"
echo "VNC is availble at <hostip>:8080/vnc.html or via VNC client on port 5900"

## https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#echo $SCRIPT_DIR
#$SCRIPT_DIR/docker_mlgl/print_jupyter.sh $docker_image_name
