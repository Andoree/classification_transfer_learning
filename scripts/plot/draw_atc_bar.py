import os
from argparse import ArgumentParser
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

POSITIONS = (0, 2, 4)
ATC_CODE_COLOR_MAPPING = {"A": "#d73027",
                          "B": "#fdae61",
                          "C": "#fee090",
                          "D": "#e0f3f8",
                          "G": "#4575b4",
                          "H": "#542788",
                          "J": "#b2abd2",
                          "L": "#bf812d",
                          "M": "#a6dba0",
                          "N": "#01665e",
                          "P": "#c51b7d",
                          "R": "#f1b6da",
                          "S": "#80cdc1",
                          "V": "#ffffbf"}

ATC_CODE_VERBOSE_MAPPING = {"A": "Alimentary tract and metabolism",
                            "B": "Blood and blood-forming organs",
                            "C": "Cardiovascular system",
                            "D": "Dermatologicals",
                            "G": "Genitourinary system\nand sex hormone",
                            "H": "Systemic hormonal preparations,\nexcluding sex hormones and insulins",
                            "J": "Anti-infectives for systemic use",
                            "L": "Antineoplastic and\nimmunomodulating agents",
                            "M": "Musculoskeletal system",
                            "N": "Nervous system",
                            "P": "Antiparasitic products,\ninsecticides, and repellents",
                            "R": "Respiratory system",
                            "S": "Sensory organs",
                            "V": "Various"}

WIDTH = 1.0
DATASETS_ORDER = ("Train", "Dev", "Test")


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default="../../data/smm4h_21_data/post_eval/ru")

    parser.add_argument('--output_path', default=r"plots/atc_bar/ru_train_subsets.pdf")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    pass
    possible_atc_code_letters = {"A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"}
    data_dict = {}
    for filename in os.listdir(input_dir):
        counter = {}
        data_path = os.path.join(input_dir, filename)
        dataset_name = filename.split('.')[0]
        dataset_name = f"{dataset_name[0].upper()}{dataset_name[1:]}"

        data_df = pd.read_csv(data_path, sep='\t')
        dataframe_columns_set = set(data_df.columns)
        present_atc_codes = possible_atc_code_letters.intersection(dataframe_columns_set)
        for atc_code in present_atc_codes:
            atc_code_num_samples = data_df[data_df[atc_code] == 1].shape[0]
            counter[atc_code] = atc_code_num_samples
        data_dict[dataset_name] = counter
    for k, v in data_dict.items():
        print(k, v)
    plt.figure(figsize=(10, 8), )
    for i, dataset_name in enumerate(DATASETS_ORDER):
        atc_statistics = data_dict[dataset_name]
        position = POSITIONS[i]
        height = sum(atc_statistics.values())
        for atc_code in sorted(atc_statistics.keys()):
            num_atc_samples = atc_statistics[atc_code]
            print(atc_code, num_atc_samples)
            # height = sum_atc_statistics - stats_accum
            # print(sum_atc_statistics, stats_accum, height)
            color = ATC_CODE_COLOR_MAPPING[atc_code]
            atc_verbose = ATC_CODE_VERBOSE_MAPPING[atc_code]
            plt.bar(position, height / sum(atc_statistics.values()), WIDTH, label=atc_verbose, color=color, )
            # stats_accum += height
            height -= num_atc_samples
    plt.xticks(POSITIONS, DATASETS_ORDER, fontsize=14, )
    plt.xticks(rotation=70)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="ATC drug groups", fontsize=13, title_fontsize="xx-large")

    plt.yticks([1, 0, 0.2, 0.4, 0.6, 0.8])
    plt.xlim([-1, 11])

    plot_file_format = output_path.split('.')[-1]
    plt.savefig(output_path, format=plot_file_format, bbox_inches='tight', )

    plt.show()


if __name__ == '__main__':
    main()
