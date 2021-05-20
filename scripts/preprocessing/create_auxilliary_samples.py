import os
import re
from argparse import ArgumentParser

import pandas as pd
import numpy as np

def split_data_samples(data_df: pd.DataFrame) -> pd.DataFrame:
    entries_dicts_list = []
    for i, row in data_df.iterrows():
        row_dict = row.to_dict()
        tweet_drug_ids_str = row.drug_id
        if tweet_drug_ids_str is not np.nan:
            tweet_drug_ids_list = re.split(rf'[+~]', tweet_drug_ids_str)
            for tweet_drug_id in tweet_drug_ids_list:
                entry_dict = {key: value for key, value in row_dict.items()}
                entry_dict["drug_id"] = tweet_drug_id
                entries_dicts_list.append(entry_dict)
        else:
            entries_dicts_list.append(row_dict)
    new_data_df = pd.DataFrame(entries_dicts_list)

    return new_data_df


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default="../../data/smm4h_2020_data/en/w_smiles/test.tsv")
    parser.add_argument('--output_path', default="../../data/smm4h_2020_data/en/w_smiles_split/test.tsv")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    data_df = pd.read_csv(input_path, sep='\t')
    print(f"Old data size: {data_df.shape[0]}")
    new_data_df = split_data_samples(data_df=data_df)
    print(f"New data size: {new_data_df.shape[0]}")
    new_data_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
