#!/bin/bash
#
# Author:

set -e
set -x


# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

# Assign variable name for label candidate we want (categories, links, keywork, etc)
project=(food technology media)

label_types=(h_cat key_phrases key_words lda links)
article_label_csv=(article_hierarchical_categories.csv article_keyphrases.csv article_keywords.csv article_lda_labels.csv article_links.csv)
label_name_csv=(hierarchical_category_names.csv keyphrases_names.csv keyword_names.csv lda_label_names.csv link_names.csv)



for topic in {0..0}
do

      # Step 1: Initialize
      exp_id=0067 #$(get_experiment_id)
      exp_dir=$(prepare_experiment_dir ${project[$topic]} ${exp_id})
      write_experiment_params ${exp_dir} weight 1

#      python -m beta.xy_embed.umap_embed \
#              --map_directory ${exp_dir} \
#              --vectors ${exp_dir}/vanilla_vectors.csv \
#              --output_file /original_xy_embeddings.csv
#
#      for i in {0..4}
#      do
#          python -m beta.joint_cluster_builder \
#              --experiment ${exp_dir} \
#              --vectors ${exp_dir}/vanilla_vectors.csv \
#              --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
#              --articles_to_labels data/food/${article_label_csv[$i]} \
#              --tf_idf_score_file ${exp_dir}/tf_idf_score.csv \
#              --loss_weight 0.5 \
#              --output_file ${exp_dir}/${label_types[$i]}_cluster_groups.csv
#
#
#
#          python -m beta.xy_embed.umap_embed \
#              --map_directory ${exp_dir} \
#              --vectors ${exp_dir}/vanilla_vectors.csv \
#              --output_file /new_xy_embeddings.csv \
#              --cluster ${exp_dir}/${label_types[$i]}_cluster_groups.csv
#
#          # label selector
#          python -m beta.regression.top_tfidf_selector \
#              --experiment ${exp_dir} \
#              --articles_to_labels data/${project[$topic]}/${article_label_csv[$x]} \
#              --percentile 1 \
#              --cluster_groups /${label_types[$i]}_cluster_groups.csv \
#              --label_name data/${project[$topic]}/${label_name_csv[$x]}\
#              --label_source ${label_types[$i]} \
#              --output_file ${label_types[$i]}_top_labels.csv
#      done

      python -m beta.label_selector ${exp_dir}

      python -m beta.borda_selector \
              --experiment ${exp_dir}\
              --borda_file predicted_borda.csv

      # draw boundary
      python -m beta.border_creator \
              ${exp_dir} /new_xy_embeddings.csv \
              /h_cat_cluster_groups.csv

      # Step 6: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
      python -m beta.json_generator \
          --map_directory data/${project} \
          --experiment ${exp_dir} \
          --filter_method kk  \
          --country_labels /final_labels.csv \
          --cluster_groups /h_cat_cluster_groups.csv \
          --embeddings /new_xy_embeddings.csv \
          --output_name /domain.json \
          --purpose experiment \
          --label_path NA


      # Step 7: Run evaluation metrics and generate HTML & SVG

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
      python -m beta.evaluation.xy_embedding_validation ${exp_dir} /new_xy_embeddings.csv >>${exp_dir}/evaluation.json
      python -m beta.evaluation.modularity_evaluator \
          --experiment ${exp_dir} \
          --xy_embeddings_csv ${exp_dir}/new_xy_embeddings.csv \
          --method nn \
          --cluster_groups_csv ${exp_dir}/h_cat_cluster_groups.csv >> ${exp_dir}/evaluation.json
      #    python -m beta.evaluation.cluster_validation_metrics \
      #            --experiment ${exp_dir} \
      #            --vectors ${exp_dir}/vanilla_vectors.csv \
      #            --groups ${exp_dir}/cluster_groups.csv >> ${exp_dir}/evaluation.json


      # Step 8: Output the html for visualization
      python -m beta.html_generator \
         --experiment ${exp_dir} \
         --purpose experiment \
         --label_path NA

done
