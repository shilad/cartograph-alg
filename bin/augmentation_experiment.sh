#!/bin/bash
#
# Augment label, pass to k-means algorithm (clustering algorithm is aware of labels)
# Augment countries (cluster), pass to tsne (xy embedding is aware of clustering)
#
# Run this script using ./bin/augmentation-experiment.sh
#
# Author: Yuren "Rock" Pang, Shilad Sen

set -e
set -x

topic=internet

# Assign variable name for label candidate we want (categories, links, keywork, etc)
article_label_csv=article_keywords.csv
label_name_csv=keyword_names.csv

# the vector we would like to use
initial_vector_for_clustering=(vanilla_vectors.csv label_augmented_vectors.csv)
vector_format_for_embedding=(vanilla_vectors.csv cluster_augmented_vectors.csv)

# xy_embedding files: tsne or umap
embed_method=umap
xy_embeddings=(xy_embeddings.csv)
cluster_groups=(cluster_groups.csv)
country_labels=(country_labels.csv)

for i in {0..1}
do
    # Step 0: Import the experiment utilities functions
    source ./bin/experiment-utils.sh

    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=$(get_experiment_id)

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir ${topic} ${exp_id})

    # Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
    # The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
    write_experiment_params ${exp_dir} num_clusters 8 \
    labels keywords \
    xy_embedding ${embed_method} \
    cluster_method kmeans

    if [[ $i == 1 ]]; then
        write_experiment_params ${exp_dir} vectors augmented
        # Step 4: Run algorithmic steps that are necessary.
        python -m cartograph.vector_augmenter \
                --experiment ${exp_dir} \
                --vectors ${exp_dir}/vanilla_vectors.csv \
                --label_vectors data/${topic}/${article_label_csv} \
                --method label
                #--output_file ${initial_vector_for_clustering[$i]}
    else
        write_experiment_params ${exp_dir} vectors vanilla
    fi

    python -m cartograph.cluster_builder \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/${initial_vector_for_clustering[$i]} \
        --clustering kmeans \
        --k 8

    python -m cartograph.label_selector \
        --experiment ${exp_dir} \
        --articles_to_labels data/${topic}/${article_label_csv} \
        --label_names data/${topic}/${label_name_csv} \
        --percentile 0.3 \
        --label_score tfidf
#        --use_candidates cluster_groups.csv \
#        --num_candidates



     # Step 4(2)(b): If you needed to generate augmented vectors
    if [[ $i == 1 ]]; then
        python -m cartograph.vector_augmenter \
                --experiment ${exp_dir} \
                --vectors ${exp_dir}/vanilla_vectors.csv \
                --label_vectors data/${topic}/${article_label_csv} \
                --method cluster
#                --cluster_vectors ${exp_dir}/cluster_groups.csv \
#                --output_file ${vector_format_for_embedding[$i]}
    fi

    # Step 5a Carry out embedding in 2D
    if [[ ${embed_method} == tsne ]] ; then
        python -m cartograph.xy_embed.tsne_embed \
                --experiment ${exp_dir} \
                --vectors ${vector_format_for_embedding[$i]}

    elif [[ ${embed_method} == umap ]] ; then
        python -m cartograph.xy_embed.umap_embed \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/${vector_format_for_embedding[$i]} \
            --clusters ${exp_dir}/${cluster_groups[0]}
    fi

    # 5b Draw boundary
    python -m cartograph.border_creator \
            --experiment ${exp_dir} \
            --xy_embeddings ${xy_embeddings[0]} \
            --cluster_groups ${cluster_groups[0]}

    # Step 6: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
    python -m cartograph.json_generator \
            --map_directory data/${topic} \
            --experiment ${exp_dir} \
            --filter_method noise
#            --country_labels ${country_labels[0]}  \
#            --cluster_groups ${cluster_groups[0]}  \
#            --xy_embeddings ${xy_embeddings[0]}   \
#            --output_file domain.json


    # Step 7: Run evaluation metrics and generate HTML & SVG
    python -m cartograph.svg_generator \
            --experiment ${exp_dir}  \
            --width 1500 \
            --height 1500 \
            --color_palette hls

    python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} >>${exp_dir}/evaluation.json

    python -m cartograph.evaluation.modularity_evaluator \
            --experiment ${exp_dir} \
            --method nn  >> ${exp_dir}/evaluation.json

    python -m cartograph.evaluation.cluster_validation_metrics \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --cluster_A ${exp_dir}/cluster_groups.csv >> ${exp_dir}/evaluation.json


    # Step 8: Output the html for visualization
    python -m cartograph.html_generator \
           --experiment ${exp_dir}

done
