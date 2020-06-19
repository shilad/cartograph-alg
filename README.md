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
* PyCharm Ultimate
* iGraph. Instructions on OS-X:
    * brew install cairo
    * brew install pkg-config
    * brew install igraph
    * On OS-X Mojave, Shilad needed to run `sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /`

## IDE Setup

* Clone this repo
* In the repo, run git lfs install
* Launch PyCharm, select "open" and the cloned directory.
* In preferences, go to the project settings, add interpreter -> New virtualenv environment with default settings.
* Open requirements.txt. PyCharm will ask you to install everything. Do it. It will take some time.
* Maybe necessary, but we aren't sure: Mark some of the top-level directories as src directories.

## Creating and updating maps

1. Add a new shell script for creating map input data. 
See `bin/make-food-map.sh` for an example template. Run the script.

2. Make sure you have git LFS installed (see above).
 Check in your new map data files, which should live in data/map_name .
 
3. If you need to update the files in the future, change your shell script, re-run it, and check in the new files.  


## Running experiments

1. Create an experiment script, similar to example-experiment.sh
2. Run the script (e.g. `./bin/run-joint.sh`.
3. The script will tell you where it sends output.
4. Only check in "final" experiments.

## Before cartograph-server

1. You need to locate your `experiment` folder and copy `domain_concept.csv` and `popularity_score.csv` from `data/[you topic]` to `experiment/[your topic]/[your experiment]`.
2. Copy this experiment folder to [catograph-server](https://github.com/shilad/cartograph-server) under the `data` directory. See instruction at [catograph-server](https://github.com/shilad/cartograph-server) to proceed.
