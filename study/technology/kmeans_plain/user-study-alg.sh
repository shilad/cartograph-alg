#!/bin/bash
#
# Generates maps for the user study that do not use augmentation.
#
# source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/augmentation-experiment.sh
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
projects=(food internet media technology)


# Assign variable name for label candidate we want (categories, links, keyword, etc)
label_types=(h_cat key_phrases key_words lda links)
article_label_csv=(article_hierarchical_categories.csv article_keyphrases.csv article_keywords.csv article_lda_labels.csv article_links.csv)
label_name_csv=(hierarchical_category_names.csv keyphrases_names.csv keyword_names.csv lda_label_names.csv link_names.csv)
label_score=tfidf


# the vector we would like to use
initial_vector_for_clustering=vanilla_vectors.csv
vector_format_for_embedding=vanilla_vectors.csv


# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=kmeans_plain

# Prepares an experiment directory
# Returns the name of the experiment directory for the map
#
prepare_experiment_dir () {
    map_name=$1

    exp_dir=study/${map_name}/${exp_id}

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

for i in {0..3}
do

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir ${projects[$i]})


    ## Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
    write_experiment_params ${exp_dir} vectors vanilla num_clusters 7 xy_embedding tsne percentile 1 labeling ${label_score}


    # Step 5 Create clusters and embedding
    python -m cartograph.cluster_builder \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/${initial_vector_for_clustering} \
        --clustering kmeans \
        --k 7

    python -m cartograph.xy_embed.tsne_embed \
            --experiment ${exp_dir} \
           --vectors ${exp_dir}/${vector_format_for_embedding}


    for x in {0..4}
    do
        label_path=${exp_dir}/labels/${label_types[$x]}

        python -m cartograph.label_selector \
            --experiment ${exp_dir} \
            --articles_to_labels data/${projects[$i]}/${article_label_csv[$x]} \
            --label_names data/${projects[$i]}/${label_name_csv[$x]} \
            --label_score ${label_score} \
            --percentile 1 \
            --label_path ${label_path}

        # Step 6: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
        python -m cartograph.json_generator data/${projects[$i]} ${exp_dir} noise ${label_path}


        # Step 7: Run evaluation metrics and generate HTML & SVG
        python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted ${label_path}

        python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} >>${label_path}/evaluation.json

        python -m cartograph.evaluation.modularity_evaluator \
                --experiment ${exp_dir} \
                --xy_embeddings_csv ${exp_dir}/xy_embeddings.csv \
                --method nn \
                --cluster_groups_csv ${exp_dir}/cluster_groups.csv >> ${label_path}/evaluation.json

        python -m cartograph.evaluation.cluster_validation_metrics \
                --experiment ${exp_dir} \
                --vectors ${exp_dir}/vanilla_vectors.csv \
                --groups ${exp_dir}/cluster_groups.csv >> ${label_path}/evaluation.json


        # Step 8: Output the html for visualization
        python -m cartograph.html_generator ${exp_dir} ${label_path}
    done
done