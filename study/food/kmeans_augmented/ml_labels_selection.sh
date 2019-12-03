# Generate data for machine learning algorithms to take in.
# Author: Lu Li
# run using ./bin/ml-labels.sh
set -e
set -x
# Assign wikiproject for data directory

projects=(food internet technology)

algs=('kmeans_augmented' 'kmeans_plain' 'LDA')

# Assign variable name for label candidate we want (categories, links, keyword, etc)
label_types=(h_cat key_phrases key_words lda links)
article_label_csv=(article_hierarchical_categories.csv article_keyphrases.csv article_keywords.csv article_lda_labels.csv article_links.csv)
label_name_csv=(hierarchical_category_names.csv keyphrases_names.csv keyword_names.csv lda_label_names.csv link_names.csv)
label_score=tfidf

# the vector we would like to use
initial_vector_for_clustering=label_augmented_vectors.csv
vector_format_for_embedding=cluster_augmented_vectors.csv

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

# Step 1: Get the experiment id.
# An experiment id can be used for multiple maps.
exp_id=(kmeans_augmented kmeans_plain LDA)


for j in {0..2}
do

  prepare_experiment_dir () {
      map_name=$1

      exp_dir=study/${map_name}/${exp_id[$j]}
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


  for i in {0..2}
  # looping through the projects
  do
      # Step 2: Prepare an experiment directory for a specific map.
      exp_dir=$(prepare_experiment_dir ${projects[$i]})

      for x in {0..4}
      # looping through the label sources

      do
          label_path=${exp_dir}/labels/${label_types[$x]}

#           Step 6 label selection
      python -m study.data.new_label_selector \
              --experiment ${exp_dir} \
              --articles_to_labels data/${projects[$i]}/${article_label_csv[$x]} \
              --label_names data/${projects[$i]}/${label_name_csv[$x]} \
              --cluster_groups ${exp_dir}/cluster_groups.csv \
              --output_file ${exp_dir}/final_labels.csv \
              --label_source ${label_types[$x]}
      done
      python -m study.data.generate_final_labels ${exp_dir} ${exp_id[$j]}

  done
done