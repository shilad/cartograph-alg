#!/bin/bash
#
# Example shell script to run an experiment
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/run-experiment.sh
#
# Author: Shilad Sen

# Import the experiment utilities functions
source ./bin/experiment-utils.sh

set -e

# Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=$(get_experiment_id)

# Prepare a map
exp_dir=$(prepare_experiment_dir food ${exp_id} spread 0.05)


#python -m cartograph.xy_embeddings_builder data/food
#python -m cartograph.cluster_builder data/food
#python -m cartograph.label_selector data/food
#python -m cartograph.json_generator data/food
#python -m cartograph.svg_generator data/food 1500 1500 muted

