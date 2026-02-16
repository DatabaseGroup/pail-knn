#!/bin/bash

# Source global configuration variables
source global.sh

export LABEL="partition"
export ALGORITHM="partition"
export MODE="sample"

. run_partition.sh
