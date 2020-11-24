import codecs
import os
from argparse import ArgumentParser

import pandas as pd


def get_unique_drug_ids(drug_ids_values):
    drug_ids_set = set()
    for drugs_str in drug_ids_values:
        drugs_list = drugs_str.split('~')
        drug_ids_set.update(drugs_list)
    return drug_ids_set


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default=r"../../data/ru_tweets_w_drugs/test.tsv")
    parser.add_argument('--output_path', default=r"english/ru_stats.txt")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    input_fname = os.path.basename(input_path)

    data_df = pd.read_csv(input_path, sep='\t')
    num_mapped_tweets = data_df.shape[0]
    drug_ids_values = data_df.drug_id
    drugs_set = get_unique_drug_ids(drug_ids_values)

    with codecs.open(output_path, 'a+') as output_file:
        output_file.write(f"{input_fname},{num_mapped_tweets},{len(drugs_set)}\n")


if __name__ == '__main__':
    main()
