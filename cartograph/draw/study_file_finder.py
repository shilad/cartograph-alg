"""
Author: Rock Pang

When there are too many experiments, we need to a way to keep track of what map we are exactly looking at

This script creates an interface to visualize a map of certain
1) project, 2) cluster number, 3) number of h_cat, and 4) simple/complex algorithms
(simple/complex algorithms mean if we adopt the 'sum' when picking the final label)

To run it, make sure install flask and jinja in your project
Also make sure line 53 points to your experiments directory
"""

from flask import Flask, render_template, request
import logging, os, re

app = Flask(__name__, template_folder='../../template')


@app.route('/')
def home():
    return render_template("study.html")


@app.route("/test", methods=["GET", "POST"])
def forward():
    mode = request.form.get('simple-complex')
    cluster = request.form.get('cluster')
    num_h_cat = request.form.get('num_h_cat')
    project = request.form.get('project')

    directory = find_file(mode, cluster, num_h_cat, project)
    body = match_body(directory)
    return render_template("study.html", cartograph=body)


def match_body(directory):
    with open(directory) as f:
        ans = ""
        isStart = False
        for line in f.readlines():
            if isStart:
                ans += line
            if '<body>' in line:
                isStart = True
            if '</body>' in line:
                isStart = False
                break
        return ans


def find_file(mode, cluster, num_h_cat, project):
    root_dir = "/Users/senresearchlab/PycharmProjects/cartograph-alg/experiments"
    cleanr = re.compile('<.*?>')
    directory = ""

    for subdir, dirs, files in os.walk(root_dir):
        topic_directory = os.path.basename(subdir)
        if project != topic_directory: continue
        for subdir, dirs, files in os.walk(subdir):
            for file in files:
                if file.endswith('.html'):
                    file = subdir + "/" + file
                    with open(file, "r") as f:
                        lines = f.readlines()
                        mode_html = re.sub(cleanr, '', lines[56]).strip()
                        num_h_cat_html = re.sub(cleanr, '', lines[61]).strip()
                        cluster_html = re.sub(cleanr, '', lines[66]).strip()
                        if mode == mode_html and num_h_cat == num_h_cat_html and cluster == cluster_html:
                            directory = file
    if len(directory) == 0:
        logging.warning("Couldn't find an experiment. Please run it on the shell script.")
        return
    return directory


if __name__ == '__main__':
    app.run(debug=True)
