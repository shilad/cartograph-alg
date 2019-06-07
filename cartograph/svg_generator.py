"""
Given a json file of articles, construct the svg visualization map

Author: Yuren "Rock" Pang
"""
import drawSvg as draw
import json
import random
import statistics
import os
import sys

colors = {}
XY_RATIO = 7
FONT_RATION = 10
CIRCLE_RATIO = 10


def scale(data):
    """
    Given the data json file, scale the popularity score using min-max scaling
    :param data:
    :return:
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


def get_articles_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
        articles = scale(data)
        return articles


def get_color(country):
    global colors
    if country in colors.keys():
        return colors[country]
    else:
        random_color = "%06x" % random.randint(0, 0xFFFFFF)
        html_color = "#" + random_color
        colors.update({country: html_color})
        return html_color


def get_country_labels_xy(articles):
    """
    Given the country json values, calculate the position of the country label
    Currently implement the median (x, y) values of each country
    :param articles:
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


def draw_country_labels(d, country_labels_xy, font_size):
    for country, position in country_labels_xy.items():
        x, y = position[0] * XY_RATIO, position[1] * XY_RATIO

        d.append(draw.Text(str(country), font_size, x, y, fill=get_color(country)))  #k is the country


def create_svg_file(directory, d):
    d.setPixelScale(2)  # Set number of pixels per geometry unit
    d.saveSvg(directory + '/graph.svg')


def draw_svg(json_articles, width, height, font_threshold=0.05*FONT_RATION, country_font_size=20):
    """
    Given a cleaned articles (popularity score scaled) as json, calculate the country label positions and draw the entire svg
    :param json_articles:
    :param width:
    :param height:
    :param font_threshold:
    :param country_font_size:
    :return:
    """
    d = draw.Drawing(width, height, origin="center")

    for v in json_articles.values():
        # Draw each article
        title = v["Article"]
        x = v["x"] * XY_RATIO
        y = v["y"] * XY_RATIO  # The original x, y are too small to visualize
        country = v["Country"]
        size = v["Popularity"] * FONT_RATION  # Augment the font_size and circle size correspondingly
        color = get_color(country)
        d.append(draw.Circle(x, y, size, fill=color))
        if size > font_threshold:
            d.append(draw.Text(title, int(size), x, y))

    # Draw country labels
    country_labels_xy = get_country_labels_xy(json_articles)
    draw_country_labels(d, country_labels_xy, country_font_size)

    return d

#
# map_directory = "../data/food"
# width = 1000
# height = 1000
# articles = get_articles_json(map_directory + "/domain.json")
# d = draw_svg(articles, float(width), float(height))
# create_svg_file(map_directory, d)

def main(map_directory, width, height):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    articles = get_articles_json(map_directory + "/domain.json")
    d = draw_svg(articles, float(width), float(height))
    create_svg_file(map_directory, d)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory, width, height = sys.argv[1:]
    main(map_directory, width, height)

