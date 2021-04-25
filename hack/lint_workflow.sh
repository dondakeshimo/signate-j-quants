#!/bin/bash

pip install 'yapf==0.31.0' 'isort==5.8.0'

lintyapf="$(yapf -dr ./src)"
lintort="$(isort --check --diff ./src)"

result=$lintyapf$lintort

if [ -z "$result" ]; then
    echo "nothing to change"
else
    echo "$lintyapf"
    echo "$lintort"
    exit 1
fi
