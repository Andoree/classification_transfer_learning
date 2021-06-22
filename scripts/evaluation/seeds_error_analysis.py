import codecs
import os
from argparse import ArgumentParser
from typing import List

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def count_sample_prediction_errors(row: pd.Series, prediction_columns_list: List, true_label_column: str = "class"):
    true_class_value = row[true_label_column]
    num_errors = 0
    for col in prediction_columns_list:
        pred_class_value = row[col]
        assert type(true_class_value) == type(pred_class_value)
        if pred_class_value != true_class_value:
            num_errors += 1
    return num_errors


def main():
    parser = ArgumentParser()
    parser.add_argument('--prediction_dir',
                        default=r"post_smm4h_21/smm4h_21_en_custom_test/exp_True_5__molbert_drugs_random_upsampling_3.0_train_13908")
    parser.add_argument('--data_path', default=r"../../data/smm4h_21_data/en_21/test.tsv")
    parser.add_argument('--pred_labels_fname', default=r"test_labels.txt")
    parser.add_argument('--output_path', default=r"../../errors/en_21/en_21_custom_molbert.tsv")
    args = parser.parse_args()

    prediction_dir = args.prediction_dir
    data_path = args.data_path
    pred_labels_fname = args.pred_labels_fname
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    keep_columns = ["tweet_id", "tweet", "class", "drug_en_name", "drug_id"]
    original_data_df = pd.read_csv(data_path, sep='\t', )

    seed_columns = []
    for name in os.listdir(prediction_dir):
        if name.startswith('seed'):
            seed = name.split('_')[-1]
            seed_prediction_column = f"pred_label_{seed}"
            seed_columns.append(seed_prediction_column)
            print(seed)

            pred_labels_path = os.path.join(prediction_dir, name, pred_labels_fname)
            pred_labels_df = pd.read_csv(pred_labels_path, header=None, names=[seed_prediction_column])[
                seed_prediction_column]
            original_data_df[seed_prediction_column] = pred_labels_df
    keep_columns.extend(seed_columns)
    keep_columns.append("pred_sum")

    original_data_df["pred_sum"] = original_data_df[seed_columns].sum(axis=1)
    original_data_df = original_data_df[keep_columns]
    # original_data_df["num_errors"] = original_data_df[original_data_df["class"] != original_data_df[seed_columns]].sum(
    #     axis=1)
    original_data_df["num_errors"] = original_data_df.apply(
        lambda row: count_sample_prediction_errors(row, prediction_columns_list=seed_columns,
                                                   true_label_column="class", ), axis=1)
    print(original_data_df.head(n=10))
    original_data_df = original_data_df[original_data_df["num_errors"] != 0]
    original_data_df.sort_values("num_errors", inplace=True, ascending=False)
    print(f"Errors dataframe size: {original_data_df.shape[0]}")
    # original_data_df.to_csv(output_path, sep='\t', index=False)
    # original_data_df["true_pred_delta"]
    # print(original_data_df.head())
    original_data_df.to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    main()
