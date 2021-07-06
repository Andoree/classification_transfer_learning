import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default="../../data/smm4h_21_data/en_21_dev_test/train.tsv")
    parser.add_argument('--output_dir', default="../../data/smm4h_21_data/en_21_dev_test_by_group/")
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    data_df = pd.read_csv(input_path, sep='\t', )
    lst = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'L', 'M', 'N', 'P', 'R', 'S', 'V']
    overall_num_ones = data_df[data_df["class"] == 1].shape[0]
    overall_num_samples = data_df.shape[0]
    overall_ones_proportion = overall_num_ones / overall_num_samples
    overall_zeros_proportion = 1. - overall_ones_proportion
    overall_zeros_ones_fraction = overall_zeros_proportion / overall_ones_proportion
    print("overall_ones_proportion:", overall_ones_proportion)
    print("overall_zeros_proportion:", overall_zeros_proportion)
    print("overall_zeros_ones_fraction", overall_zeros_ones_fraction)
    for letter in lst:
        # print('--')
        group_df = data_df[data_df[letter] == 1]
        group_num_samples = group_df.shape[0]
        positive_group_df = group_df[group_df["class"] == 1]
        negative_group_df = group_df[group_df["class"] == 0]
        group_desired_num_ones = group_num_samples * overall_ones_proportion
        group_max_possible_num_ones = positive_group_df.shape[0]
        group_num_sampled_ones = int(min(group_desired_num_ones, group_max_possible_num_ones))
        group_sampled_ones_df = positive_group_df.sample(n=group_num_sampled_ones, random_state=42)
        # print('sampled ones', group_sampled_ones_df.shape)
        group_num_sampled_zeros = int(
            min(min(group_desired_num_ones, group_max_possible_num_ones) * overall_zeros_ones_fraction,
                negative_group_df.shape[0]))
        print(letter, 'Num samples:', group_df.shape[0], 'pos:', positive_group_df.shape[0], 'neg:',
             negative_group_df.shape[0], "pos_prop", positive_group_df.shape[0] / group_df.shape[0])
        # print('aaa', group_num_sampled_zeros, negative_group_df.shape[0])
        group_samples_zeros_df = negative_group_df.sample(n=group_num_sampled_zeros, random_state=42)
        group_all_samples_df = pd.concat((group_sampled_ones_df, group_samples_zeros_df)).sample(frac=1.0,
                                                                                                 random_state=42)
        # print(group_all_samples_df.shape)
        #print(letter)
        check_ones = group_all_samples_df[group_all_samples_df["class"] == 1].shape[0]
        check_zeros = group_all_samples_df[group_all_samples_df["class"] == 0].shape[0]
        if group_all_samples_df.shape[0] > 0:
            #print(check_ones, check_zeros, check_ones / group_all_samples_df.shape[0])
            output_group_dir = os.path.join(output_dir, letter, )
            if not os.path.exists(output_group_dir):
                os.makedirs(output_group_dir)
            output_group_path = os.path.join(output_group_dir, "train.tsv")
            group_all_samples_df.to_csv(output_group_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
