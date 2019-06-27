#!/bin/bash
#
# Augment label, pass to k-means algorithm (clustering algorithm is aware of labels)
# Augment countries (cluster), pass to tsne (xy embedding is aware of clustering)
#
# Run this script using ./bin/example-experiment.sh
#
# Author: Shilad Sen

set -e
set -x

# Assign variable name for label candidate we want (categories, links, keywork, etc)
article_label_csv=article_keywords.csv
label_name_csv=keyword_names.csv

# the vector we would like to use
initial_vector_for_clustering=label_augmented_vectors.csv
vector_format_for_embedding=cluster_augmented_vectors.csv



# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=$(get_experiment_id)

# Step 2: Prepare an experiment directory for a specific map.
exp_dir=$(prepare_experiment_dir food ${exp_id})

# Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
# The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
write_experiment_params ${exp_dir} num_clusters 8 labels hierarchical xy_embedding tsne


# Step 4: Run algorithmic steps that are necessary.
python -m cartograph.vector_augmenter \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --label_vectors data/food/${article_label_csv} \
        --method label \
        --output_file ${initial_vector_for_clustering}
python -m cartograph.cluster_builder \
    --experiment ${exp_dir} \
    --vectors ${exp_dir}/${initial_vector_for_clustering} \
    --clustering kmeans \
    --k 8 \
python -m cartograph.label_selector \
    --experiment ${exp_dir} \
    --articles_to_labels data/food/${article_label_csv} \
    --label_names data/food/${label_name_csv} \
    --percentile 0.3


# Step 4(2)(b): If you needed to generate augmented vectors,
python -m cartograph.vector_augmenter \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --label_vectors data/food/article_labels.csv \
        --method cluster \
        --cluster_vectors ${exp_dir}/cluster_groups.csv \
        --output_file ${vector_format_for_embedding}

# Step 5
python -m cartograph.xy_embed.tsne_embed \
        --experiment ${exp_dir} \
       --vectors ${exp_dir}/${vector_format_for_embedding}


# Step 6: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
python -m cartograph.json_generator data/food ${exp_dir} noise


# Step 7: Run evaluation metrics and generate HTML & SVG

python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted
python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} >>${exp_dir}/evaluation.json
python -m cartograph.evaluation.modularity_evaluator \
        --experiment ${exp_dir} \
        --xy_embeddings_csv ${exp_dir}/xy_embeddings.csv \
        --method nn \
        --cluster_groups_csv ${exp_dir}/cluster_groups.csv >> ${exp_dir}/evaluation.json
python -m cartograph.evaluation.cluster_validation_metrics \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --groups ${exp_dir}/cluster_groups.csv >> ${exp_dir}/evaluation.json


# Step 8: Output the html for visualization
python -m cartograph.html_generator ${exp_dir}

