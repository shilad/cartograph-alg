

#!/bin/bash
#
# Example shell script to run an experiment
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/example-experiment.sh
#
# Author: Shilad Sen

set -e
set -x

listVar="0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.05 0.05 0.05 0.05 0.05
       0.05 0.05 0.05 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
       0.1 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.3 0.3 0.3 0.3 0.3 0.3 0.3
       0.3 0.3 0.3 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.5 0.5 0.5
       0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.7 0.7 0.7 0.7
       0.7 0.7 0.7 0.7 0.7 0.7 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8
       0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 1 1 1 1 1 1 1 1 1 1"
for i in $listVar
do

    # Step 0: Import the experiment utilities functions
    source ./bin/experiment-utils.sh


    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=0077

    # $(get_experiment_id)

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir food ${exp_id})

    # Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
    # The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
    write_experiment_params ${exp_dir} weight $i
    # Step 4: If you needed to generate augmented vectors,
    # do so now from vanilla_vectors.csv in the experiment directory.

    # Step 5: Run algorithmic steps that are necessary.

     python -m umap+kmeans.joint_umap_kmeans \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --k 8 \
            --weight $i \
            --num_neighbors 20

    # Step 6: Generate JSON
    #python -m cartograph.json_generator data/food ${exp_dir} kk  /original_country_labels.csv /original_cluster_groups.csv /original_xy_embeddings.csv /original_domain.json
    #python -m cartograph.json_generator data/food ${exp_dir} kk /country_labels.csv /y_cluster_groups.csv /y_xy_embeddings.csv /y_domain.json
    #python -m cartograph.json_generator data/food ${exp_dir} kk /country_labels.csv /y_2_cluster_groups.csv /y_2_xy_embeddings.csv /y_2_domain.json


    # Step 7: Run evaluation metrics and generate HTML & SVG
    #python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /original_domain.json /original_graph.svg /original_country_labels.csv
    #python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /new_domain.json /new_graph.svg /country_labels.csv


    # Step 8: Write evaluation metrics

    # echo "{'id' : }"
    python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /original_xy_embeddings.csv >>${exp_dir}/trustworthiness_evaluation.json
    python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /y_xy_embeddings.csv >>${exp_dir}/trustworthiness_evaluation.json
    python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /y_2_xy_embeddings.csv >>${exp_dir}/trustworthiness_evaluation.json


    python -m cartograph.evaluation.cluster_validation_metrics --experiment ${exp_dir} \
                                                               --vectors ${exp_dir}/vanilla_vectors.csv \
                                                               --groups ${exp_dir}/y_cluster_groups.csv >>${exp_dir}/cluster_evaluation.json

    python -m cartograph.evaluation.cluster_validation_metrics --experiment ${exp_dir} \
                                                               --vectors ${exp_dir}/vanilla_vectors.csv \
                                                               --groups ${exp_dir}/y_2_cluster_groups.csv >>${exp_dir}/cluster_evaluation.json

    python -m cartograph.evaluation.modularity_evaluator --experiment ${exp_dir} \
                                                         --xy_embeddings_csv ${exp_dir}/original_xy_embeddings.csv \
                                                         --method nn \
                                                         --cluster_groups_csv ${exp_dir}/original_cluster_groups.csv >>${exp_dir}/mod_evaluation.json

    python -m cartograph.evaluation.modularity_evaluator --experiment ${exp_dir} \
                                                         --xy_embeddings_csv ${exp_dir}/y_xy_embeddings.csv \
                                                         --method nn \
                                                         --cluster_groups_csv ${exp_dir}/y_cluster_groups.csv >>${exp_dir}/mod_evaluation.json

    python -m cartograph.evaluation.modularity_evaluator --experiment ${exp_dir} \
                                                         --xy_embeddings_csv ${exp_dir}/y_2_xy_embeddings.csv \
                                                         --method nn \
                                                         --cluster_groups_csv ${exp_dir}/y_2_cluster_groups.csv >>${exp_dir}/mod_evaluation.json


    # python -m cartograph.html_generator ${exp_dir}

    done