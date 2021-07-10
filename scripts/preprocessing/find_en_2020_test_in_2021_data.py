import os
from argparse import ArgumentParser

import pandas as pd


def get_2020_tweet_label(tweet_id, annotated_2021_mapping):
    if annotated_2021_mapping.get(tweet_id) is not None:
        label = annotated_2021_mapping[tweet_id]
        print(f"found: {label}")
    else:
        # print(f"No label: {tweet_id}")
        label = -1
    return label


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_2021_dir', default=r"../../data/smm4h_21_data/en/raw")
    parser.add_argument('--test_2020_path', default=r"../../data/smm4h_2020_data/en/raw/test.tsv")
    parser.add_argument('--output_path', default=r"../../data/smm4h_2020_data/en/raw/new_test.tsv")
    args = parser.parse_args()

    input_2021_dir = args.input_2021_dir
    test_2020_path = args.test_2020_path
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    test_2020_df = pd.read_csv(test_2020_path, sep='\t')
    dfs_2021_list = []
    for filename in os.listdir(input_2021_dir):
        input_path = os.path.join(input_2021_dir, filename)
        data_df = pd.read_csv(input_path, sep="\t", encoding="utf-8", quoting=3)
        replace_map = {
            "NoADE": 0,
            "ADE": 1
        }
        data_df["class"] = data_df["label"].replace(replace_map)
        dfs_2021_list.append(data_df)
    annotated_2021_df = pd.concat(dfs_2021_list, )[["tweet_id", "class"]]
    print(annotated_2021_df)
    annotated_2021_df.set_index("tweet_id", inplace=True)
    annotated_2021_df = annotated_2021_df.squeeze()
    print("2021 mapping:\n", annotated_2021_df)
    test_2020_df["class"] = test_2020_df.tweet_id.apply(lambda x: get_2020_tweet_label(x, annotated_2021_df))
    print(test_2020_df)
    test_2020_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
