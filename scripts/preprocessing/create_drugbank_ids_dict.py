import codecs
import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_data_dirs', nargs='+', default=["../../data/smm4h_21_data/post_eval/ruen",
                                                                 "../../data/smm4h_21_data/en_new/w_smiles"])
    parser.add_argument('--drugbank_path', default=r"../../data/drugbank_database.csv")
    parser.add_argument('--output_path', default="../../data/additional_data/drugs.txt")
    args = parser.parse_args()

    input_data_dirs = args.input_data_dirs
    drugbank_path = args.drugbank_path
    output_path = args.output_path

    drugbank_df = pd.read_csv(drugbank_path)
    drugbank_id_smiles_df = drugbank_df[["drugbank_id", "smiles"]]
    drugbank_id_smiles_df.set_index("drugbank_id", inplace=True)
    drugbank_id_smiles_df = drugbank_id_smiles_df.squeeze()

    present_drugbank_ids = set()
    for input_dir in input_data_dirs:
        for data_file in os.listdir(input_dir):
            input_data_path = os.path.join(input_dir, data_file)
            print(input_data_path)
            data_df = pd.read_csv(input_data_path, sep='\t')

            for i, drugbank_ids in enumerate(data_df.drug_id.values):
                if drugbank_ids is not np.nan:
                    for drug_id in re.split(rf'[+~]', drugbank_ids):
                        present_drugbank_ids.add(drug_id)
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for drug_id in sorted(present_drugbank_ids):
            out_file.write(f"{drug_id.strip()}\n")


if __name__ == '__main__':
    main()
