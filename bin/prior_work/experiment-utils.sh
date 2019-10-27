#!/usr/bin/env bash


# Finds the next unused experiment id among all experiment map directories.
#
get_experiment_id () {
    highest_free=""
    for i in $(seq 0 10000); do
        found=""
        for m in $(ls experiments); do
            d=$(printf "experiments/${m}/%04d" ${i})
            if [[ -e ${d} ]]; then
                found=true
            fi
        done
        if [[ -z ${found} ]]; then
            echo $(printf "%04d" ${i})
            return
        fi
    done
}


# Prepares an experiment directory
# Returns the name of the experiment directory for the map
#
prepare_experiment_dir () {
    map_name=$1
    exp_id=$2

    exp_dir=experiments/${map_name}/${exp_id}

    echo "using directory $exp_dir" >&2
    mkdir -p ${exp_dir}

    # Copy in the script used to run the experiment
    cp -p $0 ${exp_dir}

    # Link in the vanilla vector file
    rm ${exp_dir}/vanilla_vectors.csv >& /dev/null
    ln data/${map_name}/article_vectors.csv ${exp_dir}/vanilla_vectors.csv

    # Create empty parameters file
    touch ${exp_dir}/params.json

    echo ${exp_dir}
}


# Writes configuration parameters for the experiment to the json file.
#
write_experiment_params() {
    exp_dir=$1
    shift

    while [[ $# -gt 0 ]]; do
        param_name="$1"
        param_value="$2"
        shift 2

        echo "{ \"${param_name}\" : \"${param_value}\" }" >> ${exp_dir}/params.json
    done
}