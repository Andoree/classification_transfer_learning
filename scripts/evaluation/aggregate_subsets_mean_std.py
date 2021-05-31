import codecs
import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    # ru_tuning_drug_attention/exp_True_5_atc_drugs_sum
    parser.add_argument('--results_dir',
                        default=r"post_smm4h_21/all_features/ru_attention_exp_True_5__chemberta_drugs_molbert_drugs_rdkit_drugs_atc_drugs_random")
    # default=r"post_smm4h_21/fr_tuning_drug/exp_True_5_rdkit_rufren_drugs_random_upsampling_10.0")
    parser.add_argument('--data_path', default=r"../../data/smm4h_21_data/post_eval/ru/test.tsv")

    parser.add_argument('--output_dir', default=r"../../mean_std_scores/")
    # ru_nodrug_train_subsets.txt
    parser.add_argument('--output_fname', default=r"ru_nodrug_train_subsets.txt")
    args = parser.parse_args()

    results_dir = args.results_dir
    data_path = args.data_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_fname = args.output_fname

    # true_labels = pd.read_csv(data_path, sep='\t', )[["class", ]].values

    evaluation_results_list = []

    for train_subset_dirname in os.listdir(results_dir):
        train_size = train_subset_dirname.split('_')[-1]
        subset_dir = os.path.join(results_dir, train_subset_dirname)
        print(train_subset_dirname)
        for name in os.listdir(subset_dir):
            if name.startswith('seed'):
                seed = name.split('_')[-1]
                # print(seed)
                evaluation_file_path = os.path.join(results_dir, name, "scores.txt")
                with codecs.open(evaluation_file_path, 'r', encoding="utf-8") as eval_file:
                    for i, line in enumerate(eval_file):
                        attrs = line.strip().split(',')
                        # seed = int(attrs[0])
                        dev_p = float(attrs[1])
                        dev_r = float(attrs[2])
                        dev_f = float(attrs[3])
                        test_p = float(attrs[4])
                        test_r = float(attrs[5])
                        test_f = float(attrs[6])
                        model_dict = {
                            "train_size": train_size,
                            "seed": seed,
                            "dev_p": dev_p,
                            "dev_r": dev_r,
                            "dev_f": dev_f,
                            "test_p": test_p,
                            "test_r": test_r,
                            "test_f": test_f
                        }

                        evaluation_results_list.append(model_dict)
    results_df = pd.DataFrame(evaluation_results_list)
    print(results_df)
    print('--')
    print("Mean:")
    avg_quality_df = results_df[["dev_p", "dev_r", "dev_f", "test_p", "test_r", "test_f"]].groupby(
        by="train_size").mean()
    # print(avg_quality_df)
    print(f"--\nStd:")
    std_quality_df = results_df[["dev_p", "dev_r", "dev_f", "test_p", "test_r", "test_f"]].groupby(
        by="train_size").std()
    mean_fname = f"mean_{output_fname}"
    std_fname = f"std_{output_fname}"
    mean_output_path = os.path.join(output_dir, mean_fname)
    std_output_path = os.path.join(output_dir, std_fname)
    avg_quality_df.to_csv(mean_output_path, index=False)
    std_quality_df.to_csv(std_output_path, index=False)


if __name__ == '__main__':
    main()
