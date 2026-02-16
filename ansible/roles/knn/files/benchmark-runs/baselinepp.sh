#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="batchedlinearscanpp"
export ALGORITHM="baselinepp"
export MODE="sample"
export VECTOR="0"
export GROUPING="identity"
export MIN_BATCH_SIZE=4
export MAX_BATCH_SIZE=32

. run_prefix.sh
