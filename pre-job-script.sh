#!/bin/bash

# Step 1: List all Docker images and filter those with tags that do not end with "-latest"
images_to_remove=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v ":latest$")

# Step 2: Loop through the list and remove each image
for image in $images_to_remove; do
    docker rmi $image
done