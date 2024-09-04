FROM agrobotappliedai/jetsoncontainers:v1

ARG BRANCH_NAME=main

RUN /scripts/install-ros2-packages.sh