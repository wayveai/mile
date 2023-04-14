#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <docker_image_name>"
    exit 1
fi
docker_image_name=$1

docker stop $docker_image_name >/dev/null 2>&1 || true && docker rm $docker_image_name >/dev/null 2>&1 || true
