#!/bin/bash


docker build --build-arg MAMBA_USER_ID=$(id -u) \
             --build-arg MAMBA_USER_GID=$(id -g) \
             --build-arg SSH_KEY="$(cat ~/.ssh/id_rsa)" \
             -t vul-intext-reason:journal .
docker run --gpus all \
    --env-file .env \
    -it -d -v $PWD:/workspace \
    -v /data:/data \
    -p 8891:8888 \
    --shm-size=200g --ulimit memlock=-1 --ulimit stack=67108864 \
    --name vul-intext-reason \
    vul-intext-reason:journal

# Wait for the Docker container to start
sleep 5

# Get the logs from the running Docker container
logs=$(docker logs vul-intext-reason)

# Extract the token from the logs
token=$(echo "${logs}" | grep -oP '(?<=token=)[a-z0-9]*' | head -n 1)

echo "Jupyter Notebook Token: ${token}"
