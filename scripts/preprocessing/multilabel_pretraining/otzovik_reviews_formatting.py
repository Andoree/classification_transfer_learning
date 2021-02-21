import os
from argparse import ArgumentParser
from sys import stderr

from sklearn.model_selection import train_test_split, KFold

from inception_to_json import load_sentence_texts
import pandas as pd
import random

ANNOTATION_LABELS_LIST = ["EF", "INF", "ADR", "DI", "Finding"]


def is_sentence_annotated(efficiency_annotation):
    for label in ANNOTATION_LABELS_LIST:
        if label in efficiency_annotation:
            return True
    return False


def get_sentence_dict(efficiency_annotation, review_id, sentence_id, sentence_texts,  ):
    sent_dict = {}
    sent_dict['sentences'] = sentence_texts[sentence_id]
    for label in ANNOTATION_LABELS_LIST:
        if label in efficiency_annotation:
            sent_dict[label] = 1
        else:
            sent_dict[label] = 0

    sent_dict['annotation'] = efficiency_annotation
    sent_dict['review_id'] = review_id
    sent_dict['sentence_id'] = sentence_id
    return sent_dict


def main():
    random.seed(42)
    parser = ArgumentParser()
    parser.add_argument("--reviews_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    reviews_dir = args.reviews_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sentences = []

    for review_folder in os.listdir(reviews_dir):
        review_id = int(review_folder.split('.')[0])
        file_path = os.path.join(reviews_dir, review_folder, 'admin.tsv')
        if not os.path.exists(file_path):
            stderr.write(f"FILE DOES NOT EXIST: {file_path}\n")
            continue
        try:
            sentence_texts = load_sentence_texts(file_path=file_path)
            annotation_data = pd.read_csv(file_path, sep='\t', skip_blank_lines=True, comment='#', encoding='utf-8',
                                          names=['token_id', 'token_span', 'token', 'entity_id', 'entity_type',
                                                 'unknown_1',
                                                 'efficiency'], usecols=range(7), quoting=3)
            annotation_data['sentence_id'] = annotation_data.token_id.apply(lambda tid: int(tid.split('-')[0]))
            annotation_data['start'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[0]))
            annotation_data['end'] = annotation_data.token_span.apply(lambda tid: int(tid.split('-')[1]))
            review_dict = {}
            not_annotated_sentences_ids = set()
            review_has_annotated_sent_flag = False
            for row in annotation_data.iterrows():
                row_series = row[1]
                efficiency_annotation = row_series['efficiency']
                sentence_id = row_series['sentence_id']
                assert type(sentence_id) == int
                if review_dict.get(sentence_id) is None:
                    if is_sentence_annotated(efficiency_annotation=efficiency_annotation):
                        sent_dict = get_sentence_dict(efficiency_annotation=efficiency_annotation,
                                                      review_id=review_id, sentence_id=sentence_id,
                                                      sentence_texts=sentence_texts,
                                                    )
                        review_dict[sentence_id] = sent_dict
                        review_has_annotated_sent_flag = True
                    else:
                        not_annotated_sentences_ids.add(sentence_id)
            if review_has_annotated_sent_flag and len(not_annotated_sentences_ids) > 0:
                not_annot_sent_id = random.choice(tuple(not_annotated_sentences_ids))
                sent_dict = get_sentence_dict(efficiency_annotation="NEUTRAL",
                                              review_id=review_id, sentence_id=not_annot_sent_id,
                                              sentence_texts=sentence_texts,)
                review_dict[not_annot_sent_id] = sent_dict

            sentences.extend(review_dict.values())
        except pd.errors.ParserError as err:
            pass

    sentences_df = pd.DataFrame.from_records(sentences)
    sentences_df.rename(columns={"Other": "others"}, inplace=True)
    num_splits = args.n_splits
    if num_splits > 1:
        kf = KFold(n_splits=num_splits)
        for ind, (train_index, test_index) in enumerate(kf.split(sentences)):
            print(f"Creating fold {ind}")
            train_df, test_df= sentences_df.iloc[train_index,:], sentences_df.iloc[test_index, :]
            print(f"Fold {ind} created. Test indices {test_index[0]} : {test_index[1]}.")
            fold_directory = os.path.join(output_dir, f'fold_{ind}/')
            if not os.path.exists(fold_directory):
                os.makedirs(fold_directory)
            train_df, dev_df, _, _ = \
                train_test_split(train_df, train_df, test_size=0.1, random_state=42)
            train_path = os.path.join(fold_directory, 'train.csv')
            dev_path = os.path.join(fold_directory, 'dev.csv')
            test_path = os.path.join(fold_directory, 'test.csv')

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            dev_df.to_csv(dev_path, index=False)
    elif num_splits == 1:
        train_df, test_df, _, _ = train_test_split \
            (sentences_df, sentences_df, test_size=0.2, random_state=42)

        train_df, dev_df, _, _ = train_test_split \
            (train_df, train_df, test_size=0.1, random_state=42)

        train_path = os.path.join(output_dir, 'train.csv')
        dev_path = os.path.join(output_dir, 'dev.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        dev_df.to_csv(dev_path, index=False)
    else:
        raise Exception(f"Invalid n_splits: {num_splits}")



if __name__ == '__main__':
    main()
