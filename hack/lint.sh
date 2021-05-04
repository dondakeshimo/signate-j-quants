#!/bin/bash

DOCKER_RUN="docker run --rm -v $(pwd):/opt/ml dondakeshimo/signate-j-quants"

lintyapf="$($DOCKER_RUN yapf -dr ./src)"
lintort="$($DOCKER_RUN isort --check --diff ./src)"

result=$lintyapf$lintort

if [ -z "$result" ]; then
    echo "nothing to change"
else
    echo "$lintyapf"
    echo "$lintort"
fi
