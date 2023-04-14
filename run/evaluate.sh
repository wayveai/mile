#!/bin/bash
# Adapted from https://github.com/zhejz/carla-roach/ CC-BY-NC 4.0 license.

if [[ $# -ne 3 ]] ; then
    echo 'Please specify the CARLA executable path, the model weights, and the CARLA port.'
    exit 1
fi

CARLA_PATH=$1
PORT=$3

evaluate () {
  path=\"$1\"
  python -u evaluate.py --config-name evaluate carla_sh_path=${CARLA_PATH} agent.mile.ckpt=$path port=${PORT}
}

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate mile

# remove checkpoint files
rm outputs/port_${PORT}_checkpoint.txt
rm outputs/port_${PORT}_wb_run_id.txt
rm outputs/port_${PORT}_ep_stat_buffer_*.json

# Loop this script until the evaluation has finished because the CARLA server can crash.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  evaluate $2
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

echo "Bash script done."
