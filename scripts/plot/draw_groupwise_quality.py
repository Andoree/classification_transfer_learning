import codecs
import os
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
def main():
    parser = ArgumentParser()
    parser.add_argument('--data_paths', nargs='+',
                        default=[r"data_en_21_dev_test/en_smm4h_21_dev_test_by_group_nodrug.txt",
                                 r"data_en_21_dev_test/en_smm4h_21_dev_test_by_group_molbert_concat.txt",
                                 # r"data_en_21_dev_test/en_smm4h_21_dev_test_by_group_rdkit_attv3.txt"

                                 ])
    # en_smm4h_21_dev_test_by_group_rdkit_attv3.txt
    parser.add_argument('--label_prefixes', nargs='+',
                        default=[r"Text-only",
                                 r"Text + drug features",])
                                 #r"Text + drug features AttV3"])
    parser.add_argument('--keep_groups',  default=["A", "D", "N", ])
    # ["A", "D", "N", "R" ]
    parser.add_argument('--output_path', default=r"plots_en_21_dev_test/by_group/en_21_train_groups_dev_test.png")
    args = parser.parse_args()

    data_paths_list = args.data_paths
    label_prefixes = args.label_prefixes
    keep_groups = args.keep_groups
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    labels = []
    data = []
    for data_path, label_prefix in zip(data_paths_list, label_prefixes):
        with codecs.open(data_path, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                line = line.strip()
                attrs = line.split()
                group_letter = attrs[0]
                f1_scores = [float(x) for x in attrs[1:]]
                # if len(f1_scores) < 4 or f1_scores[0] == 0.0:
                #     continue
                if len(f1_scores) < 4:
                    continue
                if group_letter.upper() not in keep_groups:
                    continue
                label = f"{label_prefix}, group {group_letter.upper()}"
                labels.append(label)
                data.append(f1_scores)
    plt.figure(figsize=(10, 8))
    plt.xlabel("Used data percentage", fontsize=14)
    plt.ylabel("Test F-score", fontsize=14)
    plt.legend()
    max_y = -1.
    min_y = 10.
    for label, line_data in zip(labels, data):
        x_data = [k for k in range(25, 101, 25)]
        max_y = max(max_y, max(line_data))
        min_y = min(min_y, min(line_data))
        plt.plot(x_data, line_data, label=label,)
    # plt.xticks(('25%', '50%', '75%', '100%'))
    plt.xticks((25, 50, 75, 100), ('25%', '50%', '75%', '100%'))

    max_y = np.round(max_y, decimals=3)
    min_y = np.round(min_y, decimals=3)
    y_step = np.round(max_y - min_y, decimals=3) / 10

    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # plt.ylim((min_y - 1 * y_step, max_y + 5 * y_step))

    plt.legend()
    plt.savefig(output_path)
    plt.show()



if __name__ == '__main__':
    main()