#!/bin/bash
set -e
set -x

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh
projects=(food)
algs=('kmeans_augmented' 'kmeans_plain' 'LDA')
# Assign variable name for label candidate we want (categories, links, keyword, etc)
label_types=(h_cat key_phrases key_words lda links)
article_label_csv=(article_hierarchical_categories.csv article_keyphrases.csv article_keywords.csv article_lda_labels.csv article_links.csv)
label_name_csv=(hierarchical_category_names.csv keyphrases_names.csv keyword_names.csv lda_label_names.csv link_names.csv)
label_score=tfidf
algorithms=(kmeans_augmented kmeans_plain LDA)

# Step 1: Get the experiment id.
exp_id=$(get_experiment_id)

# Step 2: Prepare an experiment directory for a specific map.
exp_dir=$(prepare_experiment_dir food ${exp_id})

prepare_experiment_dir () {
      map_name=$1

      exp_dir=study/${map_name}/${algorithms[$j]}
      # Prepares an experiment directory
      # Returns the name of the experiment directory for the map

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



      # Step 2: Prepare an experiment directory for a specific map.
      exp_dir=$(prepare_experiment_dir ${projects[$i]})

for x in {0..4}
# looping through the label sources
do
    label_path=${exp_dir}/labels/${label_types[$x]}
    python -m study.data.new_label_selector \
        --experiment ${exp_dir} \
        --articles_to_labels data/$projects/${article_label_csv[$x]} \
        --label_names data/$projects/${label_name_csv[$x]} \
        --label_score ${label_score} \
        --percentile 1 \
        --purpose study \
        --label_path ${label_path} \
        --cluster_groups /cluster_groups.csv \
        --output_file /final_labels.csv \
        --use_label_candidates false \
        --num_candidates 0 \
        --alg ${algorithms[$j]} \
        --label_source ${label_types[$x]}
done
python -m study.data.generate_final_labels ${exp_dir} ${algorithms[$j]}




# Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
# The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
write_experiment_params ${exp_dir} weight 1
# Step 4: If you needed to generate augmented vectors,
# do so now from vanilla_vectors.csv in the experiment directory.

# Step 5: Run algorithmic steps that are necessary.

