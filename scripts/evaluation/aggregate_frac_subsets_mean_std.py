import codecs
import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np

def main():
    parser = ArgumentParser()
    # ru_tuning_drug_attention/exp_True_5_atc_drugs_sum
    parser.add_argument('--results_dir',
                        default=r"post_smm4h_21/en_21_dev_test_by_group/rdkit_attv3")
    # default=r"post_smm4h_21/fr_tuning_drug/exp_True_5_rdkit_rufren_drugs_random_upsampling_10.0")
    parser.add_argument('--data_path', default=r"../../data/smm4h_21_data/en_21_dev_test/test.tsv")

    parser.add_argument('--output_dir', default=r"../plot/data_en_21_dev_test")
    parser.add_argument('--output_fname', default=r"en_smm4h_21_dev_test_by_group_rdkit_attv3.txt")
    args = parser.parse_args()

    results_dir = args.results_dir
    data_path = args.data_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_fname = args.output_fname

    # true_labels = pd.read_csv(data_path, sep='\t', )[["class", ]].values

    evaluation_results_dict = {}
    for group_letter in os.listdir(results_dir):
        letter_dir = os.path.join(results_dir, group_letter)
        letter_dict = {}
        for train_subset_dirname in os.listdir(letter_dir):
            train_subset_dir = os.path.join(letter_dir, train_subset_dirname)
            if os.path.isdir(train_subset_dir):
                train_frac =  float(train_subset_dirname.split('_')[-1])
                seeds_f1_scores = []
                for name in os.listdir(train_subset_dir):
                    if name.startswith('seed'):
                        seed = name.split('_')[-1]

                        evaluation_file_path = os.path.join(train_subset_dir, name, "scores.txt")
                        if os.path.exists(evaluation_file_path):
                            with codecs.open(evaluation_file_path, 'r', encoding="utf-8") as eval_file:
                                for i, line in enumerate(eval_file):
                                    attrs = line.strip().split(',')
                                    test_p = float(attrs[5])
                                    test_r = float(attrs[6])
                                    test_f = float(attrs[7])
                                    assert test_f <= (test_p + test_r) / 2
                                    seeds_f1_scores.append(test_f)
                seeds_f1_scores = np.array(seeds_f1_scores)
                print(train_frac, len(seeds_f1_scores))

                mean_seed_f1 = seeds_f1_scores.mean()
                if np.isnan(mean_seed_f1):
                    mean_seed_f1 = 0.
                letter_dict[train_frac] = mean_seed_f1
                # print(letter_dict)
        #print(letter_dict)
        evaluation_results_dict[group_letter] = letter_dict

    for g, d in evaluation_results_dict.items():
        for k in sorted(d.keys()):
            print(k, d[k], end=' ')
        print()
    output_path = os.path.join(output_dir, output_fname)
    with codecs.open(output_path, 'w+' , encoding="utf-8") as out_file:
        for group_letter, f1_dict in evaluation_results_dict.items():
            res_array = [f1_dict[key] for key in sorted(f1_dict.keys())]
            out_file.write(f"{group_letter} {' '.join((str(x) for x in res_array))}\n")
        pass



if __name__ == '__main__':
    main()
