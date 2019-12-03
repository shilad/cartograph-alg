#!/bin/bash
#
# Author: Rock Pang, Lu Li.
# This shell scripts run a joint algorithm of embedding and clustering. The loss function (from joint algorithm)
# preserves semantic and visual spaces.
# It generate labels from clusters using key phrases label candidates and fetch h_cat labels from the top key phrases
set -e
set -x
# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh
topic=food
labels_num=70

for cluster in {8..8}
do
   # Step 1: Initialize
  exp_id=$(get_experiment_id)
  exp_dir=$(prepare_experiment_dir $topic ${exp_id})
  # Write down parameters that'll be on the html file
  write_experiment_params ${exp_dir} weight 1

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
    --cluster_groups /orig_cluster_groups.csv \
    --output_file /key_phrases_top_labels.csv \
    --label_source key_phrases \
    --num_top_labels ${labels_num}

  # Step 6: Fetch hierarchical categories from key phrases
  python -m cartograph.h_cat_fetcher \
      --experiment ${exp_dir}


  # Step 7: Output the stats and graph
  python -m cartograph.draw.border_creator \
          ${exp_dir} /original_xy_embeddings.csv \
          /orig_cluster_groups.csv

  python -m cartograph.draw.json_generator \
      --map_directory data/${topic} \
      --experiment ${exp_dir} \
      --filter_method kk  \
      --country_labels /final_labels.csv \
      --cluster_groups /orig_cluster_groups.csv \
      --embeddings /original_xy_embeddings.csv \
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
    python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /original_xy_embeddings.csv >>${exp_dir}/evaluation.json
    python -m cartograph.evaluation.modularity_evaluator \
            --experiment ${exp_dir} \
            --xy_embeddings_csv ${exp_dir}/original_xy_embeddings.csv \
            --method nn \
            --cluster_groups_csv ${exp_dir}/orig_cluster_groups.csv >> ${exp_dir}/evaluation.json
    python -m cartograph.draw.html_generator \
           --experiment ${exp_dir} \
           --purpose experiment \
           --label_path NA
done