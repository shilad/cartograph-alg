#!/usr/bin/env bash
# Generates the CSV necessary to run the clustering and labeling Mechanical Turk Study
#
# source ~/venv/cartograph-alg/bin/activate
#
# Author: Lily Irvin


set -e
set -x

exp_dir=study/evaluation
python -m study.evaluation.mturk_evaluation \
            --responses ${exp_dir}/cleaned_mturk_results.csv >>${exp_dir}/evaluation.json

#!/usr/bin/env bash