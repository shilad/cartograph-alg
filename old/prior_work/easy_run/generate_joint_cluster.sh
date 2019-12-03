#!/bin/bash
#
# Author:

set -e
set -x


# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

# Assign variable name for label candidate we want (categories, links, keywork, etc)
project=(food education internet food technology media)

#label_types=(h_cat key_phrases links lda key_words)
#article_label_csv=(article_hierarchical_categories.csv article_keyphrases.csv article_links.csv article_lda_labels.csv article_keywords.csv )
#label_name_csv=(hierarchical_category_names.csv keyphrases_names.csv link_names.csv lda_label_names.csv keyword_names.csv)

label_types=(key_phrases)
article_label_csv=(article_keyphrases.csv)
label_name_csv=(keyphrases_names.csv)

for topic in {0..0}
do
#
#      for cluster in {9..9}
#      do
        # Step 1: Initialize
        exp_id=0111 #$(get_experiment_id)
        exp_dir=$(prepare_experiment_dir ${project[$topic]} ${exp_id})
        write_experiment_params ${exp_dir} weight 1

        # Step 2: run UMAP
#        python -m beta.xy_embed.umap_embed \
#                --map_directory ${exp_dir} \
#                --vectors ${exp_dir}/vanilla_vectors.csv \
#                --output_file /original_xy_embeddings.csv
#
##         Step 3: start looping through five label sources
#        for i in {0..0}
#        do
#            # Step 3.a. run joint cluster algorithm
#            python -m beta.joint_cluster_builder \
#                --experiment ${exp_dir} \
#                --vectors ${exp_dir}/vanilla_vectors.csv \
#                --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
#                --articles_to_labels data/${project[$topic]}/${article_label_csv[$i]} \
#                --output_file ${exp_dir}/${label_types[$i]}_cluster_groups.csv \
#                --label_names data/${project[$topic]}/${label_name_csv[$i]} \
#                --k 8
#
#
#            # Step 3.c. get top labels, country, tfidf score, cluster algorithm, and label source.
#            python -m beta.feature_data_generator \
#              --experiment ${exp_dir} \
#              --articles_to_labels data/${project[$topic]}/${article_label_csv[$i]} \
#              --label_names data/${project[$topic]}/${label_name_csv[$i]} \
#              --cluster_groups /${label_types[$i]}_cluster_groups.csv \
#              --output_file /${label_types[$i]}_top_labels.csv \
#              --label_source ${label_types[$i]}
#        done
#
#        # Step 4: gathers feature data from all label sources, flatten them, and run the model on it.
#        python -m beta.feature_data_collector ${exp_dir}

        # Step 3.b. run UMAP
        python -m beta.xy_embed.umap_embed \
            --map_directory ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --output_file /new_xy_embeddings.csv \
            --cluster ${exp_dir}/key_phrases_cluster_groups.csv

        # Step 5: selects the label with top predicted borda as labels

        python -m beta.borda_selector \
                --experiment ${exp_dir} \
                --borda_file predicted_borda.csv

        # Step 6: draw boundary
        python -m beta.border_creator \
                ${exp_dir} /new_xy_embeddings.csv \
                /key_phrases_cluster_groups.csv

        # Step 7: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
        python -m beta.json_generator \
            --map_directory data/${project[$topic]} \
            --experiment ${exp_dir} \
            --filter_method kk  \
            --country_labels /final_labels.csv \
            --cluster_groups /key_phrases_cluster_groups.csv \
            --embeddings /new_xy_embeddings.csv \
            --output_name /domain.json \
            --purpose experiment \
            --label_path NA


        # Step 8: generate SVG

        python -m beta.svg_generator \
            --map_directory ${exp_dir} \
            --width 1500 \
            --height 1500 \
            --color_palette hls \
            --json_file /domain.json \
            --output_file /graph.svg \
            --country_labels /final_labels.csv \
            --purpose experiment \
            --label_path NA
#      done
done
