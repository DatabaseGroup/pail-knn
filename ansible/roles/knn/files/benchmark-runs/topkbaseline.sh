#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="topkbaseline"
export ALGORITHM="topkbaseline"
export MODE="sample"
export VECTOR="0"
export GROUPING="aio"

. run_prefix.sh
