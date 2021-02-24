import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ['sentences', 'EF', 'INF', 'ADR', 'DI', 'Finding']


def main():
    eng_corpus_dir = r"../../../data/med_reviews_corpora/psytar_csvs"
    otzovik_path = r"../../../data/med_reviews_corpora/full_otzovik_csv/full_otzovik.csv"
    output_dir = "../../../data/med_reviews_corpora/merged_reviews"
    corpus = 'cadec'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    random_state = 42
    train_frac = 0.9

    # otzovik_path = os.path.join(otzovik_dir, "train.csv")
    # dev_otzovik_path = os.path.join(otzovik_dir, "dev.csv")
    otzovik_df = pd.read_csv(otzovik_path, encoding="utf-8")[COLUMNS]
    otzovik_df = otzovik_df.sample(frac=1.0)
    otzovik_num_samples = otzovik_df.shape[0]
    print("Otzovik num samples:", otzovik_num_samples)
    otzovik_num_train_samples = int(otzovik_num_samples * train_frac)
    # otzovik_num_dev_samples = otzovik_num_samples - otzovik_num_train_samples
    otzovik_train_df = otzovik_df[:otzovik_num_train_samples]
    otzovik_dev_df = otzovik_df[otzovik_num_train_samples:]
    print("Otzovik train:", otzovik_train_df.shape[0])
    print("Otzovik dev:", otzovik_dev_df.shape[0])
    print("ADR", otzovik_df[otzovik_df.ADR == 1].shape[0])
    print("EF", otzovik_df[otzovik_df.EF == 1].shape[0])
    print("INF", otzovik_df[otzovik_df.INF == 1].shape[0])
    print("DI", otzovik_df[otzovik_df.DI == 1].shape[0])
    print("Finding", otzovik_df[otzovik_df.Finding == 1].shape[0])
    print("Train stats:")
    print("ADR", otzovik_train_df[otzovik_train_df.ADR == 1].shape[0])
    print("EF", otzovik_train_df[otzovik_train_df.EF == 1].shape[0])
    print("INF", otzovik_train_df[otzovik_train_df.INF == 1].shape[0])
    print("DI", otzovik_train_df[otzovik_train_df.DI == 1].shape[0])
    print("Finding", otzovik_train_df[otzovik_train_df.Finding == 1].shape[0])
    print(otzovik_dev_df[otzovik_dev_df.ADR == 1]["sentences"].values)

    if corpus == 'psytar':
        psytar_train_df = pd.read_csv(os.path.join(eng_corpus_dir, "train.csv"), encoding="utf-8")[COLUMNS]
        psytar_dev_df = pd.read_csv(os.path.join(eng_corpus_dir, "dev.csv"), encoding="utf-8")[COLUMNS]

        merged_train_df = pd.concat([psytar_train_df, otzovik_train_df]).sample(frac=1, random_state=random_state)
        #merged_dev_df = pd.concat([psytar_dev_df, otzovik_dev_df]).sample(frac=1, random_state=random_state)

        merged_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", index=False)
        otzovik_dev_df.to_csv(os.path.join(output_dir, "dev.csv"), encoding="utf-8", index=False)
    elif corpus == 'cadec':
        cadec_train_df = pd.read_csv(os.path.join(eng_corpus_dir, "train.csv"), encoding="utf-8")[COLUMNS]
        merged_train_df = pd.concat([cadec_train_df, otzovik_train_df]).sample(frac=1, random_state=random_state)
        merged_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", index=False)
        otzovik_dev_df.to_csv(os.path.join(output_dir, "dev.csv"), encoding="utf-8", index=False)
    else:
        raise Exception(f"Unsupported corpus: {corpus}")


if __name__ == '__main__':
    main()
