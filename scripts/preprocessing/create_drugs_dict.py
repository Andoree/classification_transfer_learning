import codecs
import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import pandas as pd

from scripts.preprocessing.map_en_tweets_to_drugbank import load_drugbank_dict
from scripts.preprocessing.map_ru_tweets_to_drugbank import get_drugbank_list, Drug


def get_drug_tokens_from_drugs_list(drugs_list: List[Drug]) -> List[str]:
    """
    :param drugs_list: List of "Drug" class instances
    :return: List of possible drug mentions
    """
    tokens = []
    for drug in drugs_list:
        if drug.name is not np.nan:
            tokens.append(drug.name)

        if drug.compound is not np.nan:
            tokens.append(drug.compound)

        for drug_selected_term in drug.selected_terms:
            tokens.append(drug_selected_term)
    return tokens


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_drugbank_path', default=r"../../data/drugbank_aliases.json")
    # r"../../df_all_terms_ru_en.csv", r"../../data/drugbank_aliases.json"
    parser.add_argument('--language', default=r"en")
    parser.add_argument('--output_path', default=r"../../data/en_drug_tokens.txt")
    args = parser.parse_args()

    input_drugbank_path = args.input_drugbank_path
    language = args.language

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    if language == "ru":
        drug_name_column = "Drug name in Russian"
        selected_terms_column = "selected_terms_w_cases"
        compound_column = "Active compound in Russian"
    elif language == "en":
        drug_name_column = "Drug"
        selected_terms_column = "target_med"
        compound_column = "normalized_med"
    else:
        raise ValueError(f"Invalid language: {language}")
    if language == "ru":
        drugbank_df = pd.read_csv(input_drugbank_path, )
        drugbank_list = get_drugbank_list(drugbank_df, drug_name_column=drug_name_column,
                                          selected_terms_column=selected_terms_column,
                                          active_compound_column=compound_column)
        drug_tokens_list = get_drug_tokens_from_drugs_list(drugs_list=drugbank_list)
    elif language == "en":
        drugbank_dict = load_drugbank_dict(input_drugbank_path)
        drug_tokens_list = list(drugbank_dict.keys())
    else:
        raise ValueError(f"Unsupported language: {language}")
    drug_tokens_list = set(drug_tokens_list)
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for drug_token in drug_tokens_list:
            out_file.write(f"{drug_token.strip()}\n")


if __name__ == '__main__':
    main()
