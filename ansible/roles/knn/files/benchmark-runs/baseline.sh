#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="linearscan"
export ALGORITHM="baseline"
export MODE="sample"
export VECTOR="0"
export GROUPING="identity"

. run_prefix.sh
