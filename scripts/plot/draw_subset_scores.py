import os
import sys
from argparse import ArgumentParser

import numpy
import pandas as pd
from matplotlib import pyplot as plt


def draw_plots(data, xlabel, ylabel, plot_labels=None, filename='pplot.png',
               x_numeric=True, rotate_x=False, top_offset=4, bot_offset=3, line_style=None,
               yaxis_unit_segment=None, xaxis_unit_segment=None, scatter=False, figsize=None, title=None,
               marker_size=None, format='png'):
    if figsize is not None:
        plt.figure(figsize=figsize, )
    colors = ['g', 'blue', 'red']
    markers = ['o', 'v', 'd']
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    max_x = 0  # Максимальное чисто топиков
    min_x = sys.maxsize  # Минимальное число топиков

    max_y = -sys.maxsize  # Минимальная оценка
    min_y = sys.maxsize  # Минимальная оценка
    # markers = itertools.cycle((',', '+', '.', 'o', '*', 'v', 'x'))
    lines = []
    for i in range(len(data)):
        marker = markers[i // 3]
        color = colors[i % 3]
        CB_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00', ]
        x = data[i][0]
        y = data[i][1]
        if x_numeric:
            max_x = max(x) if max(x) > max_x else max_x
            min_x = min(x) if min(x) < min_x else min_x

        max_y = max(y) if max(y) > max_y else max_y
        min_y = min(y) if min(y) < min_y else min_y
        label = plot_labels[i] if plot_labels is not None else None

        if scatter:
            plt.scatter(x, y, label=label, marker=next(markers), )
        else:
            #     line_style = LINE_STYLES[i]
            line = plt.plot(x, y, label=label, marker=marker, linestyle=line_style, color=CB_colors[i % 3],
                            markersize=marker_size, markevery=0.05)
            print(line[0].get_color())
            lines.append(line)
    if plot_labels is not None:
        plt.legend()

    max_y = numpy.round(max_y, decimals=3)
    min_y = numpy.round(min_y, decimals=3)
    y_step = numpy.round(max_y - min_y, decimals=3) / 10
    if rotate_x:
        plt.xticks(rotation=-90)
    if x_numeric:
        # Настройка единичных отрезков графика
        plt.xticks(numpy.arange(min_x, max_x + 1, 5.0))
    if yaxis_unit_segment is not None:
        yticks = [i for i in range(int(round(min_y, - 1) + 1), int(round(max_y, - 1)))
                  if i % yaxis_unit_segment == 0]
        yticks.extend((max_y, min_y))
        plt.yticks(yticks)

    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.ylim((min_y - bot_offset * y_step, max_y + top_offset * y_step))
    plt.title(title)

    plt.savefig(filename, format=format, bbox_inches='tight', )

    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_paths', nargs='+',
                        default=[r"../evaluation/mean_std_scores/mean_ru_nodrug_train_subsets.txt",
                                 r"../evaluation/mean_std_scores/mean_ru_drug_train_subsets.txt"])

    parser.add_argument('--output_path', default=r"ru_train_subsets.txt")
    args = parser.parse_args()

    data_path_list = args.data_paths
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    data_list = []

    for data_path in data_path_list:
        data_df = pd.read_csv(data_path)
        train_sizes = data_df.train_size.values
        test_f1_scores = data_df.test_f.values
        data_list.append((train_sizes, test_f1_scores))
    for i, (x, y) in enumerate(data_list):
        line = plt.plot(x, y, label=str(i), )
    plt.show()


if __name__ == '__main__':
    main()
