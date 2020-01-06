#!/bin/bash
#
# Author: Rock Pang
# This shell runs joint algorithms for user study
# variables:
# 1) simple - complex, 2) cluster, 3) h_cat from key phrases 4) topics
#
#

set -e
set -x
# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh
topicArray=(food education georgraphy)
labels_num_array=(70 80 95 105 120 180)
isSumInKeyPhraseArray=(True False)

for cluster in {7..10}
do
  for topic in "${topicArray[@]}"
  do
    for isSumInKeyPhrase in "${isSumInKeyPhraseArray[@]}"
    do
      for labels_num in "${labels_num_array[@]}"
      do
          # Step 1: Initialize
          exp_id=$(get_experiment_id)
          exp_dir=$(prepare_experiment_dir ${topic} ${exp_id})
          # Write down parameters that'll be on the html file
          write_experiment_params ${exp_dir} hasSum ${isSumInKeyPhrase} top_key_phrase_label_number ${labels_num} cluster_num ${cluster} \
                                  low_weight 0.05 label_weight 0.08

          # Step 2: run UMAP
          python -m cartograph.xy_embed.umap_embed \
            --map_directory ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --output_file /original_xy_embeddings.csv

          # Step 3: run joint cluster algorithm
          python -m cartograph.cluster.joint_cluster_builder \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
            --articles_to_labels data/${topic}/article_keyphrases.csv \
            --output_file ${exp_dir}/key_phrases_cluster_groups.csv \
            --label_names data/$topic/keyphrases_names.csv \
            --k $cluster

          # Step 4: get top labels, country, tfidf score, cluster algorithm, and label source.
          python -m cartograph.top_label_source_generator \
            --experiment ${exp_dir} \
            --articles_to_labels data/$topic/article_keyphrases.csv \
            --label_names data/$topic/keyphrases_names.csv \
            --cluster_groups /key_phrases_cluster_groups.csv \
            --output_file /key_phrases_top_labels.csv \
            --label_source key_phrases \
            --num_top_labels ${labels_num}

           # Step 5: UMAP (re-generate embedding based on clusters)
          python -m cartograph.xy_embed.umap_embed \
            --map_directory ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --output_file /new_xy_embeddings.csv \
            --cluster ${exp_dir}/key_phrases_cluster_groups.csv

          # Step 6: Fetch hierarchical categories from key phrases
          python -m cartograph.h_cat_fetcher \
              --experiment ${exp_dir} \
              --isSumInKeyPhrase ${isSumInKeyPhrase}


          # Step 7: Output the stats and graph
          python -m cartograph.draw.border_creator \
                  ${exp_dir} /new_xy_embeddings.csv \
                  /key_phrases_cluster_groups.csv
          python -m cartograph.draw.json_generator \
              --map_directory data/${topic} \
              --experiment ${exp_dir} \
              --filter_method kk  \
              --country_labels /final_labels.csv \
              --cluster_groups /key_phrases_cluster_groups.csv \
              --embeddings /new_xy_embeddings.csv \
              --output_name /domain.json \
              --purpose experiment \
              --label_path NA
          python -m cartograph.draw.svg_generator \
              --map_directory ${exp_dir} \
              --width 1500 \
              --height 1500 \
              --color_palette hls \
              --json_file /domain.json \
              --output_file /graph.svg \
              --country_labels /final_labels.csv \
              --purpose experiment \
              --label_path NA
            python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /new_xy_embeddings.csv >>${exp_dir}/evaluation.json
            python -m cartograph.evaluation.modularity_evaluator \
                    --experiment ${exp_dir} \
                    --xy_embeddings_csv ${exp_dir}/new_xy_embeddings.csv \
                    --method nn \
                    --cluster_groups_csv ${exp_dir}/key_phrases_cluster_groups.csv >> ${exp_dir}/evaluation.json
            python -m cartograph.draw.html_generator \
                   --experiment ${exp_dir} \
                   --purpose experiment \
                   --label_path NA
          done
      done
  done
done