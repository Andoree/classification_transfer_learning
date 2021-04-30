import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from typing import Dict, Set

import numpy as np
import pandas as pd


def get_atc_codes_first_char_values_set(atc_codes_list):
    chars_set = set()
    for drug_atcs in atc_codes_list:
        for atc_code in drug_atcs:
            first_char = atc_code[0]
            chars_set.add(first_char)
    return chars_set


def get_atc_codes_by_drug_ids(data_df: pd.DataFrame, possible_atc_codes: Set[str], id_to_atc_mapping: Dict[str, str],
                              drugs_sep: str = '~', ) -> str:
    entries = []
    for drug_ids_str in data_df["drug_id"].values:
        atc_codes_dict = {code: 0 for code in possible_atc_codes}
        if drug_ids_str is np.nan:
            entries.append(atc_codes_dict)
            continue
        drugs_list = re.split(rf'[+{drugs_sep}]', drug_ids_str)
        for drug_id in drugs_list:
            drug_atc_codes = id_to_atc_mapping[drug_id]
            for atc_code in drug_atc_codes:
                atc_first_char = atc_code[0]
                assert atc_first_char in possible_atc_codes
                atc_codes_dict[atc_first_char] = 1
        entries.append(atc_codes_dict)

    atc_features_df = pd.DataFrame(entries)
    return atc_features_df



def main():
    parser = ArgumentParser()
    parser.add_argument('--input_drugbank', default=r"../../data/drugbank_database.csv")
    parser.add_argument('--input_tweets', default=r"../../data/smm4h_21_data/post_eval/ruen/dev.tsv")
    # "../../data/smm4h_all/smm4h_2021_preprocessed_w_drugs/test.tsv")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/post_eval/ruen_w_atc/dev.tsv")
    # r"../../data/smm4h_all/smm4h_2021_preprocessed_w_smiles/test.tsv")
    args = parser.parse_args()

    drugbank_path = args.input_drugbank
    tweets_path = args.input_tweets
    seed = args.seed
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    drugbank_df = pd.read_csv(drugbank_path, converters={"atc_codes": literal_eval})
    drugbank_id_atc_code_df = drugbank_df[["drugbank_id", "atc_codes"]]
    tweets_df = pd.read_csv(tweets_path, sep='\t', )
    drugbank_id_atc_code_df.set_index("drugbank_id", inplace=True)
    possible_atc_first_chars_set = get_atc_codes_first_char_values_set(drugbank_df["atc_codes"].values)

    drugbank_id_atc_code_df = drugbank_id_atc_code_df.squeeze()
    atc_features_df =  get_atc_codes_by_drug_ids(tweets_df, possible_atc_first_chars_set, drugbank_id_atc_code_df)
    test_delete_df = tweets_df.loc[:, "A": "V", ]
    print(test_delete_df)
    tweets_df = pd.concat((tweets_df, atc_features_df),axis=1)

    tweets_df.to_csv(output_path, sep='\t', index=False,)


if __name__ == '__main__':
    main()
