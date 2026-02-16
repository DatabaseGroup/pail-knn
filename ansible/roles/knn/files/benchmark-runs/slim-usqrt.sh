#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="slim-usqrt"
export ALGORITHM="slim"
export MODE="sample"
export VECTOR="0"
export GROUPING="usqrt"

. run_prefix.sh
