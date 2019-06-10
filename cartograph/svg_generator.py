"""
Given a json file of articles, construct the svg visualization map

Author: Yuren "Rock" Pang
"""

# wrap titles (multi-line), make title size as big as largest city label,
# create classes of font sizes (first 15 cities get ____ size, next 30 get
# ____ size, etc.


import drawSvg as draw
import json
import pandas as pd
import statistics
import seaborn as sns
from collections import defaultdict
import operator


XY_RATIO = 7
FONT_RATIO = 10


def get_sizes(articles):
    sizes = defaultdict(lambda: .5)
    cities = {}
    for value in articles.values():
        cities[value['Article']] = value['Popularity']
    x = 0
    for key, value in sorted(cities.items(), key=operator.itemgetter(1), reverse=True):
        if x == 0:
            sizes[key] = 30
            x += 1
        elif x <= 15:
            sizes[key] = 15
            x += 1
        elif x <= 45:
            sizes[key] = 8
            x += 1
        elif x <= 135:
            sizes[key] = 2
            x += 1
    return sizes


def get_articles_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
        articles = scale(data)
    return articles


def scale(data):
    """
    Given the data json file, scale the popularity score using min-max scaling
    """
    max = 0    # smallest popularity score is 0
    min = sys.maxsize

    for v in data.values():
        popularity_score = float(v["Popularity"])
        if popularity_score > max:
            max = popularity_score
        if popularity_score < min:
            min = popularity_score

    denominator = max - min

    for k, v in data.items():
        temp = v["Popularity"]
        v["Popularity"] = (temp - min)/denominator
    return data


def set_colors(countries_csv, color_palette):
    countries = pd.read_csv(countries_csv)
    colors = {}
    palette = sns.color_palette(color_palette, len(countries)).as_hex()
    for i in range(len(countries['country'])):
        colors[countries.iloc[i, 1]] = palette[i]
    return colors


def draw_svg(json_articles, width, height, colors, sizes, country_font_size=30):
    """
    Given a json of cleaned articles (popularity score scaled), calculate the country label positions and draw the entire svg
    """
    drawing = draw.Drawing(width, height, origin="center")

    for v in json_articles.values():
        # Draw each article
        title = v["Article"]
        x = v["x"] * XY_RATIO
        y = v["y"] * XY_RATIO  # The original x, y are too small to visualize
        country = v["Country"]
        size = sizes[v['Article']]  # Augment the font_size and circle size correspondingly
        drawing.append(draw.Circle(x, y, size, fill=colors[country]))
        if size > .5:
            drawing.append(draw.Text(title, int(size), x, y))

    # Draw country labels
    country_labels_xy = get_country_labels_xy(json_articles)
    draw_country_labels(drawing, country_labels_xy, country_font_size, colors)
    return drawing


def get_country_labels_xy(articles):
    """
    Given the country json values, calculate the position of the country label
    Currently implement the median (x, y) values of each country
    :return: dict, key: country (str), value: list[x][y] (floats)
    """
    country_x_y_list = {}
    country_labels = {}

    for key, value in articles.items():
        country = value["Country"]
        x, y = value["x"], value["y"]

        if country not in country_x_y_list.keys():
            country_x_y_list[country] = [[], []]

        country_x_y_list[country][0].append(x)
        country_x_y_list[country][1].append(y)

    for k, v in country_x_y_list.items():
        if k not in country_labels.keys():
            country_labels[k] = [[], []]

        country_labels[k][0] = statistics.median(v[0])
        country_labels[k][1] = statistics.median(v[1])

    return country_labels


def draw_country_labels(d, country_labels_xy, font_size, colors):
    for country, position in country_labels_xy.items():
        x, y = position[0] * XY_RATIO, position[1] * XY_RATIO
        d.append(draw.Text(str(country).upper(), font_size, x, y, fill=colors[country], center=True, stroke='black', stroke_width=0.4))  #k is the country


def create_svg_file(directory, d):
    d.setPixelScale(2)  # Set number of pixels per geometry unit
    d.saveSvg(directory + '/graph.svg')


def main(map_directory, width, height, color_palette):
    articles = get_articles_json(map_directory + "/domain.json")
    colors = set_colors(map_directory + "/country_labels.csv", color_palette)
    sizes = get_sizes(articles)
    drawing = draw_svg(articles, float(width), float(height), colors, sizes)
    create_svg_file(map_directory, drawing)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory, width, height, color_palette = sys.argv[1:]
    main(map_directory, width, height, color_palette)

