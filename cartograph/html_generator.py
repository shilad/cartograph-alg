"""
Given a svg file and evaluation metric, output a html for easier visualization

June 21 needs to add more parameters

Author: Yuren 'Rock' Pang
"""
import time
from collections import OrderedDict
import argparse
from jinja2 import Template
from xml.dom import minidom
import json


def get_params_and_values(file):
    params_evals = {}
    eval_json = list(json.loads(line) for line in open(file, 'r'))
    for obj in eval_json:
        dic = list(obj.items())[0]
        params_evals.update({dic[0]: dic[1]})
    return params_evals


def main(experiment_directory, evaluation_json, params_json, graph_path):
    experiment_directory = experiment_directory.rstrip('/')
    eval_dic = get_params_and_values(evaluation_json)
    params_dic = OrderedDict()
    params_dic['timestamp'] = time.ctime()
    params_dic['directory'] = experiment_directory
    params_dic.update(get_params_and_values(params_json))

    with open('./template/index.html') as file:
        template = Template(file.read())

        names = experiment_directory.split('/')

        doc = minidom.parse(graph_path)
        svg = doc.toxml()

        return template.render(title= 'Graph of ' + names[-2] + " (experiment "+ names[-1] + ")",
                               svg=svg,
                               eval_dic=eval_dic,
                               params_dic=params_dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args()

    experiment_directory = str(args.experiment)
    evaluation_json = experiment_directory + "/evaluation.json"
    params_json = experiment_directory + "/params.json"
    graph_path = experiment_directory + "/graph.svg"

    file = open(experiment_directory + "/index.html", "w+")
    file.write(main(experiment_directory, evaluation_json, params_json, graph_path))
    file.close()



# html = main("../experiments/food/0001", "../experiments/food/0001/evaluation.json", "../experiments/food/0001/params.json")
# file = open("../experiments/food/0001/index.html", "w+")
# file.write(html)
# file.close()
#
# # doc = minidom.parse("../experiments/food/0001/graph.svg")
# # print(doc.toxml())
