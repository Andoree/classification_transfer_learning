import codecs
import os
from argparse import ArgumentParser
from collections import Counter
from typing import List

import numpy as np
import pandas as pd


def count_drug_frequency(values: List[str], ):
    """
    :param values: List of strings with '~' value separator
    """
    counter = Counter()
    for val in values:
        if val is not np.nan:
            drug_id_list = val.split('~')
            counter.update(drug_id_list)
    return counter


def write_frequencies(output_path: str, counter):
    """
    :param output_path: Path to output stats file. Values
    and frequencies are written to this file
    :param counter: collections.Counter with filled value frequencies
    """
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for val, freq in counter.most_common():
            out_file.write(f"{val}\t{freq}\n")

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dataset_path', default=r"../../data/smm4h_21_data/en_new/w_smiles_all/en_21_all.tsv")
    parser.add_argument('--output_dir', default=r"drug_frequency_stats/en_2021")
    args = parser.parse_args()

    input_dataset_path = args.input_dataset_path
    # output_path = args.output_path
    # output_dir = os.path.dirname(output_path)
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    dataset_df = pd.read_csv(input_dataset_path, sep='\t', )

    drug_id_values = dataset_df.drug_id.values
    drug_id_counter = count_drug_frequency(drug_id_values)

    drug_name_values = dataset_df.drug_en_name.values
    drug_name_counter = count_drug_frequency(drug_name_values)

    drug_id_stats_path = os.path.join(output_dir, "stats_drug_id.txt")
    write_frequencies(output_path=drug_id_stats_path, counter=drug_id_counter)
    drug_name_stats_path = os.path.join(output_dir, "stats_drug_name.txt")
    write_frequencies(output_path=drug_name_stats_path, counter=drug_name_counter)



if __name__ == '__main__':
    main()
