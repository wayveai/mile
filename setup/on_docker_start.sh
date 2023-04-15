#!/bin/bash
set -e
service ssh start >/dev/null 2>&1

function setup_passwordless_ssh() {
    local user="$1"
    local authorized_keys="$2"

    if [ "${authorized_keys}" != "**None**" ]; then
        home_dir=$(eval echo ~${user})
        ssh_dir="${home_dir}/.ssh"

        mkdir -p "${ssh_dir}"
        chown ${user}:${user} "${ssh_dir}"
        chmod 700 "${ssh_dir}"

        touch "${ssh_dir}/authorized_keys"
        chown ${user}:${user} "${ssh_dir}/authorized_keys"
        chmod 600 "${ssh_dir}/authorized_keys"

        IFS=$'\n'
        arr=$(echo ${authorized_keys} | tr "," "\n")

        for x in $arr
        do
            x=$(echo $x | sed -e 's/^ *//' -e 's/ *$//')
            echo "$x" >> "${ssh_dir}/authorized_keys"
        done
    else
        echo "ERROR: No authorized keys found in \$AUTHORIZED_KEYS for user ${user}"
        exit 1
    fi
}

# Call the function for both root and carla users with the same AUTHORIZED_KEYS
setup_passwordless_ssh "root" "${AUTHORIZED_KEYS}"
setup_passwordless_ssh "carla" "${AUTHORIZED_KEYS}"

# Launch VNC
exec supervisord -c /vnc/supervisord.conf &

export DISPLAY=:0.0
env | egrep -v "^(HOME=|USER=|MAIL=|LC_ALL=|LS_COLORS=|LANG=|HOSTNAME=|PWD=|TERM=|SHLVL=|LANGUAGE=|_=)" >> /etc/environment

# wait till the display is set
# su - carla -c "sleep 2 && /home/carla/mycarla.sh" &

# jupyter-lab &
# give writing permissions to mile folder to 'carla' user
chmod g+rwx /home/carla/mile

# HYDRA_FULL_ERROR=1 bash run/evaluate.sh  ~/CarlaUE4.sh ~/mile.ckpt 2000
# Xvfb :0 -screen 0 1024x768x24 -listen tcp -ac
exec "$@"
