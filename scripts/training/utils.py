import codecs
import re
from random import randrange
from typing import Set, Dict, List

import nltk
import numpy as np
import torch
from natasha import Segmenter, Doc
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"


def mask_drug(text: str, drugs_set: Set[str], drug_mask: str = "DRUG"):
    """
    :param text: Raw tweet string
    :param drugs_set: Set of possible forms of drug mentions
    :param drug_mask: Mask to replace drug mentions with
    :return: Tweet string with masked drug mentions
    """
    ru_letters = set("абвгдеёжзийклмнопрстуфхцчъыьэюя")
    en_letters = set('abcdefghijklmnopqrstuvwxyz')
    ru_counter = 0
    en_counter = 0
    for char in text:
        if char in ru_letters:
            ru_counter += 1
        elif char in en_letters:
            en_counter += 1
    if ru_counter > en_counter:
        segmenter = Segmenter()
        natasha_doc = Doc(text)
        natasha_doc.segment(segmenter)
        tokens = [token.text for token in natasha_doc.tokens]
    else:
        tokens = nltk.word_tokenize(text)

    replace_tokens = []
    for token in tokens:
        if token.lower() in drugs_set:
            replace_tokens.append(token)
    replace_tokens.sort(key=lambda t: -len(t), )
    for token in replace_tokens:
        text = re.sub(token, drug_mask, text, flags=re.IGNORECASE)

    return text


def get_smiles_list(smiles_list, molecules_sep='~~~'):
    preprocessed_smiles = []
    for smile_str in smiles_list:
        if smile_str is np.nan:
            preprocessed_smiles.append([""])
        else:
            preprocessed_smiles.append(smile_str.split(molecules_sep))
    return preprocessed_smiles


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_labels_probas(labels_path, probas_path, labels, probas):
    with codecs.open(labels_path, 'w+', encoding="utf-8") as labels_file, \
            codecs.open(probas_path, 'w+', encoding="utf-8") as probas_file:
        for label, probability in zip(labels, probas):
            labels_file.write(f"{label}\n")
            probas_file.write(f"{probability}\n")


def write_hyperparams(apply_upsampling, positive_class_weight, n_epochs, dropout, freeze_layer_count,
                      freeze_embeddings_layer, text_model_name, output_path):
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"model name: {text_model_name}\n")
        out_file.write(f"Upsampling: {apply_upsampling}\nUpsampling_weight: {positive_class_weight}\n")
        out_file.write(f"n_epochs: {n_epochs}\ndropout: {dropout}\n")
        out_file.write(
            f"freeze_layer_count : {freeze_layer_count}\nfreeze_embeddings_layer: {freeze_embeddings_layer}\n")


def encode_smiles(model, tokenizer, smiles_list, max_length, molecules_sep='~~~'):
    model.eval()
    with torch.no_grad():
        model_hidden_size = model.config.hidden_size
        molecules_embeddings = []
        for sample in tqdm(smiles_list, mininterval=7.0):
            sample_embeddings = []
            if sample is not np.nan:
                molecules_smiles = sample.split(molecules_sep)
                for smile_str in molecules_smiles:
                    encoded_molecule = tokenizer.encode(smile_str, max_length=max_length,
                                                        padding="max_length", truncation=True, return_tensors="pt").to(
                        device)
                    output = model(encoded_molecule, return_dict=True)
                    cls_embedding = output["last_hidden_state"][0][0].cpu()
                    sample_embeddings.append(cls_embedding)
                mean_sample_embedding = torch.mean(torch.stack(sample_embeddings), dim=0)
            else:
                mean_sample_embedding = torch.zeros(size=[model_hidden_size, ], dtype=torch.float32)
            molecules_embeddings.append(mean_sample_embedding)
    return molecules_embeddings


def embedding_str_to_numpy(s):
    numbers_strs = s.strip("[]").split()
    emb_size = len(numbers_strs)
    embedding = np.empty(shape=emb_size, dtype=np.float32)
    for i in range(emb_size):
        embedding[i] = np.float(numbers_strs[i])
    return embedding


def get_sider_emb_by_drugbank_id(drugbank_ids, sider_embs, drugs_sep='~', emb_size=1320):
    if (type(drugbank_ids) == str and drugbank_ids.strip() == '') or drugbank_ids is np.nan:
        embedding = np.zeros(shape=emb_size, dtype=np.float32)
        return embedding
    drugbank_ids_list = drugbank_ids.split(drugs_sep)

    for drug_id in drugbank_ids_list:
        if drug_id not in sider_embs:
            embedding = np.zeros(shape=emb_size, dtype=np.float32)
        else:
            embedding = sider_embs[drug_id]
        if np.isnan(embedding[0]):
            continue
        else:
            return embedding
    embedding = np.zeros(shape=emb_size, dtype=np.float32)
    return embedding


def load_drugs_dict(dict_path: str) -> Set[str]:
    drugs = set()
    with codecs.open(dict_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            drugs.add(line.strip())
    return drugs


def load_drug_features(drug_features_path: str) -> Dict[str, np.array]:
    drug_features_dict = {}
    with codecs.open(drug_features_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            drugbank_id = attrs[0]
            feature_vector = [float(x) for x in attrs[1].split()]
            feature_vector = np.array(feature_vector, dtype=np.float32)
            drug_features_dict[drugbank_id] = feature_vector
    return drug_features_dict


def split_drugs_ids_str(drugbank_ids_str: str) -> List[str]:
    if drugbank_ids_str is np.nan:
        return []
    drug_ids_str = re.split(rf'[+~]', drugbank_ids_str)
    return drug_ids_str


def is_vector_zeros(vector: np.array) -> bool:
    for value in vector:
        if value != 0.0:
            return False
    return True


def sample_drug_features(drug_features_dict: Dict[str, np.array], drug_features_size: int,
                         drug_ids_list: List[str], sampling_type: str) -> np.array:
    num_drugs = len(drug_ids_list)
    if num_drugs > 0:
        if sampling_type != "mean" and sampling_type != "sum":
            if sampling_type == "random":
                perm = np.random.permutation(num_drugs)
            else:
                perm = range(num_drugs)
            for i in perm:
                drug_id = drug_ids_list[i]
                drug_features = drug_features_dict.get(drug_id)
                if drug_features is not None:
                    if not is_vector_zeros(drug_features):
                        return drug_features
        else:
            drug_features_embs = []
            for drug_id in drug_ids_list:
                drug_features = drug_features_dict.get(drug_id)
                if drug_features is not None and not is_vector_zeros(drug_features):
                    drug_features_embs.append(drug_features)
            if len(drug_features_embs) > 0:
                drug_features_embs = np.array(drug_features_embs)
                if sampling_type == "mean":
                    drug_features = drug_features_embs.mean(axis=0)
                elif sampling_type == "sum":
                    drug_features = drug_features_embs.sum(axis=0)
                else:
                    raise ValueError(f"Invalid sampling type: {sampling_type}")
                return drug_features

    drug_features = np.zeros(shape=drug_features_size, dtype=np.float32)
    return drug_features


def get_drug_text_emb(text: str, drug_mention_emb_dict: Dict[str, np.array], sampling_type, drug_features_size: int):
    ru_letters = set("абвгдеёжзийклмнопрстуфхцчъыьэюя")
    en_letters = set('abcdefghijklmnopqrstuvwxyz')
    fr_letters = set("abcdefghijklmnopqrstuvwxyzéèàùâêîôûëïüÿç")
    ru_counter = 0
    en_counter = 0
    fr_counter = 0
    for char in text:
        if char in ru_letters:
            ru_counter += 1
        if char in en_letters:
            en_counter += 1
        if char in fr_letters:
            fr_counter += 1
    if ru_counter > en_counter and ru_counter > fr_counter:
        segmenter = Segmenter()
        natasha_doc = Doc(text)
        natasha_doc.segment(segmenter)
        tokens = [token.text for token in natasha_doc.tokens]
    elif en_counter >= ru_counter and en_counter >= fr_counter:
        tokens = nltk.word_tokenize(text)
    elif fr_counter > ru_counter and fr_counter > en_counter:
        tokens = nltk.word_tokenize(text, language='french')
    else:
        raise Exception(f"Could not determine the language of the text: {text}")

    drugs_text_mentions = []
    drugs_set = set(list(drug_mention_emb_dict.keys()))
    for token in tokens:
        if token.lower() in drugs_set:
            drugs_text_mentions.append(token.lower())
    # drugs_text_mentions = np.array(drugs_text_mentions)
    num_drug_mentions = len(drugs_text_mentions)
    if num_drug_mentions > 0:
        if sampling_type != "mean" and sampling_type != "sum":
            if sampling_type == "random":
                sampled_mention_id = randrange(num_drug_mentions)
                sampled_mention_text = drugs_text_mentions[sampled_mention_id]
            else:
                sampled_mention_text = drugs_text_mentions[0]
            drug_text_emb = drug_mention_emb_dict[sampled_mention_text]
        else:
            drugs_text_mentions_embs = [drug_mention_emb_dict[token] for token in drugs_text_mentions if
                                        not is_vector_zeros(drug_mention_emb_dict[token])]
            drugs_text_mentions_embs = np.array(drugs_text_mentions_embs)
            if sampling_type == "mean":
                drug_text_emb = drugs_text_mentions_embs.mean(axis=0)
            elif sampling_type == "sum":
                drug_text_emb = drugs_text_mentions_embs.sum(axis=0)
            else:
                raise ValueError(f"Invalid sampling type: {sampling_type}")
    else:
        drug_text_emb = np.zeros(shape=drug_features_size, dtype=np.float32)

    return drug_text_emb


def encode_drug_text_mentions(drugs_strs: Set[str], max_seq_length: int, text_encoder, text_tokenizer) \
        -> Dict[str, np.array]:
    drug_str_emb_dict = {}
    text_encoder.eval()
    with torch.no_grad():
        for drug_mention in drugs_strs:
            tokenizer_output = text_tokenizer.encode(drug_mention, max_length=max_seq_length,
                                                     padding="max_length", truncation=True, return_tensors="pt").to(
                device)
            output = text_encoder(tokenizer_output, return_dict=True)
            cls_embedding = output["last_hidden_state"][0][0].cpu().numpy()
            drug_str_emb_dict[drug_mention] = cls_embedding
    return drug_str_emb_dict
