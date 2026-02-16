#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="eager"
export ALGORITHM="slim"
export MODE="sample"
export VECTOR="0"
export GROUPING="identity"

. run_prefix.sh
