# Building:

## Local:
```
docker build -t agrobotappliedai/mlcontainers-cpu:latest . --network=host
```

## DockerHub:
```
docker buildx build --platform linux/amd64,linux/arm64 -t agrobotappliedai/ ros-containers:latest -f ROS-Dev.Dockerfile . --network=host --push
```

# Running: 
```
docker run -it --rm --gpus all -v ~/Downloads:/home/usr/Downloads agrobotappliedai/ros-containers:latest --network=host
```