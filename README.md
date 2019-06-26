# Cartograph Algorithms
Algorithm development repository for Cartograph team.

Team members: Lily Irvin, Lu Li, Yuren "Rock" Pang, Jonathan Scott, Shilad Sen.

## Software dependencies

* Python 3
* Python modules in requirements.txt
* Git LFS. Instructions on OS-X:
   * brew install git-lfs
   * sudo git lfs install
   * sudo git lfs install --system
* NLTK. Instructions on OS-X:
    * pip install -U nltk
    * sudo mkdir -p /usr/local/share/nltk_data
    * sudo python -m nltk.downloader -d usr/local/share/nltk_data all
* iGraph. Instructions on OS-X:
    * brew install cairo
    * brew install pkg-config
    * brew install igraph
    * sudo pip install python-igraph
   
## Creating and updating maps

1. Add a new shell script for creating map input data. 
See `bin/make-food-map.sh` for an example template. Run the script.

2. Make sure you have git LFS installed (see above).
 Check in your new map data files, which should live in data/map_name .
 
3. If you need to update the files in the future, change your shell script, re-run it, and check in the new files.  


## Running experiments

1. Create an experiment script, similar to example-experiment.sh
2. Run the script (e.g. `./bin/example-experiment.sh`.
3. The script will tell you where it sends output.
4. Only check in "final" experiments.
