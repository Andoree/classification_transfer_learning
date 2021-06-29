import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default=r"../../data/smm4h_21_data/post_eval/ru")
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/post_eval/ru_full/all.tsv")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    train_path = os.path.join(input_dir, "train.tsv")
    dev_path = os.path.join(input_dir, "dev.tsv")
    test_path = os.path.join(input_dir, "test.tsv")

    train_df = pd.read_csv(train_path, sep='\t', )
    dev_df = pd.read_csv(dev_path, sep='\t', )
    test_df = pd.read_csv(test_path, sep='\t')
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    print(f"Sizes: \nTrain: {train_df.shape}\nDev: {dev_df.shape}\nTest: {test_df.shape}\nAll:{all_df.shape}")
    all_df.drop_duplicates(inplace=True)
    print(f"All no duplicates:{all_df.shape}")

    all_df.to_csv(output_path, sep='\t', index=False, )


if __name__ == '__main__':
    main()
