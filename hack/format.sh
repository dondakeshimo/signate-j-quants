#!/bin/bash

DOCKER_RUN="docker run --rm -v $(pwd):/opt/ml"

if [ "$(uname)" == 'Linux' ]; then
    DOCKER_RUN="$DOCKER_RUN -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -u $(id -u $USER):$(id -g $USER)"
fi

DOCKER_RUN="$DOCKER_RUN  dondakeshimo/signate-j-quants"

$DOCKER_RUN yapf -ir ./src
$DOCKER_RUN isort ./src
