import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--results_dir',
                        default=r"post_smm4h_21/fr_tuning_drug_attention_new/exp_True_5_molbert_rufren_drugs_random_upsampling_10.0")
    parser.add_argument('--data_path', default=r"../../data/smm4h_2020_data/fr/raw/test.tsv")
    parser.add_argument('--output_dir',
                        default=r"post_smm4h_21/seeds/fr_tuning_drug_attention_new/exp_True_5_molbert_rufren_drugs_random_upsampling_10.0")
    args = parser.parse_args()

    results_dir = args.results_dir
    data_path = args.data_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    final_output_dirname = output_dir.split('/')[-1]
    unlabelled_data_df = pd.read_csv(data_path, sep='\t', )

    for name in os.listdir(results_dir):
        if name.startswith('seed'):
            seed = name.split('_')[-1]
            print(seed)
            test_labels_path = os.path.join(results_dir, name, "test_labels.txt")
            test_labels_df = pd.read_csv(test_labels_path, sep="\t", encoding="utf-8", header=None, names=["labels"])
            unlabelled_data_df["Class"] = test_labels_df.labels
            seed_prediction_df = unlabelled_data_df[["tweet_id", "Class"]]
            output_seed_fname = f"pred_{final_output_dirname}_{seed}.tsv"
            output_seed_path = os.path.join(output_dir, output_seed_fname)
            seed_prediction_df.to_csv(output_seed_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
