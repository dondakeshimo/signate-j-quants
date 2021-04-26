#!/bin/bash

DOCKER_RUN="docker run --rm -v $(pwd):/opt/ml dondakeshimo/signate-j-quants"

$DOCKER_RUN yapf -ir ./src
$DOCKER_RUN isort ./src
