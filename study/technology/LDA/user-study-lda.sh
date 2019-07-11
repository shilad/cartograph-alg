#!/bin/bash
#
# Generates maps for the user study using LDA.
#
# source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/augmentation-experiment.sh
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
projects=(media science technology)


# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

for i in {0..2}
do
    echo ${projects[$i]}
    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=LDA


    # Prepares an experiment directory
    # Returns the name of the experiment directory for the map
    #
    prepare_experiment_dir () {
        map_name=${projects[$i]}
        exp_id=$1

        exp_dir=study/${map_name}/${exp_id}

        echo "using directory $exp_dir" >&2
        mkdir -p ${exp_dir}

        # Copy in the script used to run the experiment
        cp -p $0 ${exp_dir}

        echo ${exp_dir}
    }


    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir ${exp_id})


    # Step 3: Run LDA:
    python -m cartograph.topic_finder ${exp_dir} data/${projects[$i]}
done
#!/usr/bin/env bash