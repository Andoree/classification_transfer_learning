import codecs
import os
from argparse import ArgumentParser
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from scripts.preprocessing.append_atc_codes import get_atc_codes_first_char_values_set
from scripts.training.utils import load_drugs_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dict_path', default="../../data/additional_data/drugs.txt")
    # molbert/atc/text_emb/rdkit
    parser.add_argument('--features_type', default="atc")
    # cimm-kzn/enrudr-bert, roberta-large
    parser.add_argument('--text_encoder_name', default="cimm-kzn/enrudr-bert")
    parser.add_argument('--chem_encoder_name', default="./models/seyonec/ChemBERTa_zinc250k_v2_40k/model")
    parser.add_argument('--drug_encoder_max_length', type=int, default=256)
    parser.add_argument('--molbert_embs_path', default=r"../../data/additional_data/drugbank_id_molbert.csv")
    parser.add_argument('--drugbank_path', default=r"../../data/drugbank_database.csv")
    parser.add_argument('--rdkit_path', default=r"../../data/additional_data/chem_descriptors.csv")

    parser.add_argument('--output_dir', default="../../data/additional_data/features/")

    args = parser.parse_args()

    input_dict_path = args.input_dict_path
    features_type = args.features_type
    text_encoder_name = args.text_encoder_name
    chem_encoder_name = args.chem_encoder_name
    drug_encoder_max_length = args.drug_encoder_max_length
    molbert_embs_path = args.molbert_embs_path
    drugbank_path = args.drugbank_path
    rdkit_path = args.rdkit_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    device = "cuda" if torch.cuda.is_available else "cpu"
    drugbank_ids_set = load_drugs_dict(input_dict_path)

    features_fname = f"{features_type}"
    features = []
    if features_type == "chemberta":
        drugbank_df = pd.read_csv(drugbank_path)
        drugbank_id_smiles_df = drugbank_df[["drugbank_id", "smiles"]]
        drugbank_id_smiles_df.set_index("drugbank_id", inplace=True)
        drugbank_id_smiles_df = drugbank_id_smiles_df.squeeze()

        drug_tokenizer = AutoTokenizer.from_pretrained(chem_encoder_name, )
        drug_encoder = AutoModel.from_pretrained(chem_encoder_name, )
        drug_encoder.eval()
        with torch.no_grad():
            features_fname += f"_{text_encoder_name.split('/')[-1]}"
            for drugbank_id in drugbank_ids_set:
                if drugbank_id in drugbank_id_smiles_df:
                    drug_smile_str = drugbank_id_smiles_df[drugbank_id]
                    if drug_smile_str is not np.nan:
                        encoded_molecule = drug_tokenizer.encode(drug_smile_str, max_length=drug_encoder_max_length,
                                                                 padding="max_length", truncation=True,
                                                                 return_tensors="pt").to(
                            device)
                        output = drug_encoder(encoded_molecule, return_dict=True)
                        cls_embedding = output["last_hidden_state"][0][0].cpu().numpy()
                        features.append((drugbank_id, cls_embedding))
                    else:
                        print(f"No smiles: {drugbank_id}")
                else:
                    print(f"Not found drugbank id: {drugbank_id}")
    elif features_type == "molbert":
        molbert_embs_df = pd.read_csv(molbert_embs_path)

        molbert_embs_df = molbert_embs_df[["drugbank_id", "molbert_vect"]]
        molbert_embs_df.set_index("drugbank_id", inplace=True)
        molbert_embs_df = molbert_embs_df.squeeze()
        for drugbank_id in tqdm(drugbank_ids_set):
            if drugbank_id in molbert_embs_df:
                molbert_emb_str = molbert_embs_df[drugbank_id]
                molbert_emb = [float(x) for x in molbert_emb_str.split()]
                features.append((drugbank_id, molbert_emb))
            else:
                print(f"No smiles: {drugbank_id}")
    elif features_type == "atc":
        drugbank_df = pd.read_csv(drugbank_path, converters={"atc_codes": literal_eval})
        drugbank_id_atc_code_df = drugbank_df[["drugbank_id", "atc_codes"]]
        drugbank_id_atc_code_df.set_index("drugbank_id", inplace=True)

        possible_atc_first_chars_set = get_atc_codes_first_char_values_set(drugbank_df["atc_codes"].values)
        drugbank_id_atc_code_df = drugbank_id_atc_code_df.squeeze()
        for drugbank_id in drugbank_ids_set:
            if drugbank_id in drugbank_id_atc_code_df and drugbank_id is not np.nan:
                drug_atcs_list = drugbank_id_atc_code_df[drugbank_id]
                drug_features = {code: 0 for code in possible_atc_first_chars_set}
                for atc_code in drug_atcs_list:
                    atc_code_first_char = atc_code[0]
                    assert atc_code_first_char in possible_atc_first_chars_set
                    drug_features[atc_code_first_char] = 1
                drug_features = [drug_features[key] for key in sorted(drug_features.keys())]

                features.append((drugbank_id, drug_features))

    elif features_type == "rdkit":
        molbert_embs_df = pd.read_csv(molbert_embs_path)

        molbert_embs_df = molbert_embs_df[["drugbank_id", "darwin_smiles"]]
        molbert_embs_df.set_index("drugbank_id", inplace=True)
        molbert_embs_df = molbert_embs_df.squeeze()

        rdkit_df = pd.read_csv(rdkit_path)
        print(rdkit_df.shape)
        rdkit_df.drop_duplicates(inplace=True)
        print(rdkit_df.shape)
        rdkit_df.fillna(value=0.0, inplace=True)
        unique_counter = rdkit_df.nunique()
        unique_counter = unique_counter[unique_counter > 1]
        # multiclass_columns = unique_counter[(unique_counter < 10) & (unique_counter > 2)].index.tolist()
        # counter = 0
        # for col in multiclass_columns:
        #     unique_vals = rdkit_df[col].unique()
        #     counter += len(unique_vals)
        #     print(sorted(unique_vals))
        # print("Counter:", counter)

        binary_columns_candidates = unique_counter[unique_counter == 2].index.tolist()
        binary_columns = []
        for col in binary_columns_candidates:
            unique_vals = rdkit_df[col].unique()
            if 0. in unique_vals and 1. in unique_vals:
                binary_columns.append(col)
        print('binary:', len(binary_columns))
        unique_counter.sort_values(inplace=True)
        rdkit_features_columns = unique_counter.index.tolist()
        print("Feature columns:", len(rdkit_features_columns))
        rdkit_df = rdkit_df[rdkit_features_columns]
        print(rdkit_df.shape)

        rdkit_features_columns = list(rdkit_df.columns)
        rdkit_features_columns.remove("Smiles")
        rdkit_quantitative_columns = rdkit_features_columns.copy()
        for col in binary_columns_candidates:
            rdkit_quantitative_columns.remove(col)
        rdkit_df.fillna(0.0, inplace=True)

        rdkit_mean = rdkit_df[rdkit_quantitative_columns].mean(axis=0)
        rdkit_std = rdkit_df[rdkit_quantitative_columns].std(axis=0)
        rdkit_df[rdkit_quantitative_columns] = (rdkit_df[rdkit_quantitative_columns] - rdkit_mean) / rdkit_std

        for drugbank_id in drugbank_ids_set:
            if drugbank_id in molbert_embs_df:
                drug_smile_str = molbert_embs_df[drugbank_id]
                if drug_smile_str is not np.nan:
                    drug_features = rdkit_df[rdkit_df.Smiles == drug_smile_str][rdkit_features_columns].values[0]

                    if drug_features.shape[0] == 2:
                        print("Two samples:", drugbank_id, drug_smile_str)
                    features.append((drugbank_id, drug_features))
                else:
                    print(f"No smiles: {drugbank_id}, {drug_smile_str}")
            else:
                pass
                # print(f"Not found drugbank id: {drugbank_id}")

    input_fname = os.path.basename(input_dict_path)
    output_fname = f"{features_fname}_{input_fname}"
    output_path = os.path.join(output_dir, output_fname)
    print(len(features))
    print(len(features[0][1]))
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for drug_features in features:
            drug_id = drug_features[0]
            features_array = drug_features[1]
            out_file.write(f"{drug_id}\t{' '.join((str(x) for x in features_array))}\n")


if __name__ == '__main__':
    main()
