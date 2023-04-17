#!/usr/bin/env bash
set -e
PROJECT_NAME=mile

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
./setup/stop_sandbox.sh $PROJECT_NAME
# Build parent image
# ./docker_mlgl/build.sh mlgl_sandbox
docker build -t $PROJECT_NAME $SCRIPT_DIR
./setup/start_sandbox.sh $PROJECT_NAME $SCRIPT_DIR

SANDBOX_IP="$(docker inspect -f '{{ .NetworkSettings.IPAddress }}' $PROJECT_NAME)"
ssh carla@$SANDBOX_IP