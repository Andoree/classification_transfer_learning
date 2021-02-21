import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ['sentences', 'EF', 'INF', 'ADR', 'DI', 'Finding']


def main():
    eng_corpus_dir = r"../cadec/reformatted"
    otzovik_dir = r"../otzovik_csvs/fold_0"
    output_dir = "../merged_cadec_ru_otzovik/fold_0"
    corpus = 'cadec'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    random_state = 42

    train_otzovik_path = os.path.join(otzovik_dir, "train.csv")
    dev_otzovik_path = os.path.join(otzovik_dir, "dev.csv")
    otzovik_train_df = pd.read_csv(train_otzovik_path, encoding="utf-8")[COLUMNS]
    otzovik_dev_df = pd.read_csv(dev_otzovik_path, encoding="utf-8")[COLUMNS]
    if corpus == 'psytar':
        psytar_train_df = pd.read_csv(os.path.join(eng_corpus_dir, "train.csv"), encoding="utf-8")[COLUMNS]
        psytar_dev_df = pd.read_csv(os.path.join(eng_corpus_dir, "dev.csv"), encoding="utf-8")[COLUMNS]

        merged_train_df = pd.concat([psytar_train_df, otzovik_train_df]).sample(frac=1, random_state=random_state)
        merged_dev_df = pd.concat([psytar_dev_df, otzovik_dev_df]).sample(frac=1, random_state=random_state)

        merged_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", index=False)
        merged_dev_df.to_csv(os.path.join(output_dir, "dev.csv"), encoding="utf-8", index=False)
    elif corpus == 'cadec':
        cadec_train_df = pd.read_csv(os.path.join(eng_corpus_dir, "train.csv"), encoding="utf-8")[COLUMNS]
        merged_train_df = pd.concat([cadec_train_df, otzovik_train_df]).sample(frac=1, random_state=random_state)
        merged_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", index=False)
        otzovik_dev_df.to_csv(os.path.join(output_dir, "dev.csv"), encoding="utf-8", index=False)
    else:
        raise Exception(f"Unsupported corpus: {corpus}")


if __name__ == '__main__':
    main()
