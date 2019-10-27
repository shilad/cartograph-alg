# Generate the training feature dataset for machine learning algorithm to take in.
# Author: Lu Li
# run using ./bin/ml-labels.sh
set -e
set -x
# Assign wikiproject for data directory

projects=(Media food internet technology)

algs=('kmeans_augmented' 'kmeans_plain')

# Assign variable name for label candidate we want (categories, links, keyword, etc)

# the vector we would like to use
initial_vector_for_clustering=label_augmented_vectors.csv
algs=(kmeans_augmented kmeans_plain)

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


# looping through the projects
for topic in {0..3}
do
  prepare_experiment_dir () {
      map_name=$1

      exp_dir=study/feature
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
    exp_dir=$(prepare_experiment_dir ${projects[$topic]})

    # using kmeans plain to get cluster groups
#    python -m cartograph.cluster_builder \
#      --experiment ${exp_dir} \
#      --vectors ${exp_dir}/vanilla_vectors.csv \
#      --k 7 \
#      --output_file ${exp_dir}/${projects[$topic]}_kmeans_plain_cluster_groups.csv
#
#    # using kmeans augmented to get cluster groups
#    python -m cartograph.vector_augmenter \
#        --experiment ${exp_dir} \
#        --vectors ${exp_dir}/vanilla_vectors.csv \
#        --label_vectors data/${projects[$topic]}/article_labels_combined.csv \
#        --method label \
#        --output_file /${initial_vector_for_clustering}
#    python -m cartograph.cluster_builder \
#        --experiment ${exp_dir} \
#        --vectors ${exp_dir}/${initial_vector_for_clustering} \
#        --k 7 \
#        --output_file ${exp_dir}/${projects[$topic]}_kmeans_augmented_cluster_groups.csv


    # using LDA to get cluster groups
    # TODO: fix LDA

#      python -m cartograph.topic_finder ${exp_dir} data/${projects[$i]} 10 /LDA_labels.csv /article_topic_distribution.csv
#      python -m cartograph.topic_finder ${exp_dir} data/${projects[$i]} 50 /LDA_labels_50.csv /article_topic_distribution_50.csv
#      python -m study-old.data.LDA_cluster_builder ${exp_dir}/LDA_

#           Step 6 label selection
      python -m study.data.new_label_selector \
              --label_dir data/ \
              --cluster_groups ${exp_dir}/${projects[$topic]}_ \
              --output_file ${exp_dir}/top_label_tfidfs.csv

done

