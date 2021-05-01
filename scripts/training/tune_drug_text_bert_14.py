import codecs
import configparser
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from attention import BertCrossattLayer
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, RobertaModel
from transformers import AutoTokenizer
from utils import mask_drug, get_smiles_list, epoch_time, save_labels_probas, write_hyperparams, \
    load_drugs_dict, load_drug_features, split_drugs_ids_str, sample_drug_features

device = "cuda" if torch.cuda.is_available else "cpu"


class TweetsDataset(Dataset):
    def __init__(self, tweets_df, text_tokenizer, molecule_tokenizer=None, molecule_max_length=256,
                 text_max_length=128, sampling_type="first", drugs_dictionary=None,
                 drug_features_dict=None, drug_features_size=None):
        self.labels = tweets_df["class"].astype(np.float32).values
        self.text_max_length = text_max_length
        self.sampling_type = sampling_type
        self.molecule_max_length = molecule_max_length
        tweets = tweets_df.tweet.values
        if drugs_dictionary is not None:
            tweets = [mask_drug(text, drugs_set=drugs_dictionary, ) for text in tweets]
        self.tokenized_tweets = [text_tokenizer.encode_plus(x, max_length=self.text_max_length,
                                                            padding="max_length", truncation=True,
                                                            return_tensors="pt", ) for x in tweets]
        if drug_features_dict is not None and drug_features_size is not None:
            self.drugbank_ids = [split_drugs_ids_str(drug_ids_str) for drug_ids_str in tweets_df.drug_id.values]
        self.tokenized_molecules = None
        self.drug_features_dict = drug_features_dict
        self.drug_features_size = drug_features_size

        if molecule_tokenizer is not None:
            smiles_list = get_smiles_list(tweets_df.smiles.values)
            self.tokenized_molecules = [molecule_tokenizer.batch_encode_plus(x, max_length=self.molecule_max_length,
                                                                             padding="max_length", truncation=True,
                                                                             return_tensors="pt", ) for x in
                                        smiles_list]
        self.drug_embeddings = tweets_df.drug_embedding.values
        self.smiles = tweets_df.smiles.values

    def __getitem__(self, idx):
        sample_dict = {
            "input_ids": self.tokenized_tweets[idx]["input_ids"][0],
            "attention_mask": self.tokenized_tweets[idx]["attention_mask"][0],
            "drug_embeddings": self.drug_embeddings[idx],
            "labels": self.labels[idx]
        }
        if self.drug_features_dict is not None:
            drug_ids_list = self.drugbank_ids[idx]
            drug_features = sample_drug_features(drug_features_dict=self.drug_features_dict,
                                                 drug_features_size=self.drug_features_size,
                                                 drug_ids_list=drug_ids_list, sampling_type=self.sampling_type)
            sample_dict["drug_features"] = drug_features

        if self.tokenized_molecules is not None:
            if self.sampling_type == "random":
                num_samples = self.tokenized_molecules[idx]["input_ids"].size()[0]
                sample_id = random.randint(0, num_samples - 1)
                sample_dict["molecule_input_ids"] = self.tokenized_molecules[idx]["input_ids"][sample_id]
                sample_dict["molecule_attention_mask"] = self.tokenized_molecules[idx]["attention_mask"][sample_id]
            else:
                sample_dict["molecule_input_ids"] = self.tokenized_molecules[idx]["input_ids"][0]
                sample_dict["molecule_attention_mask"] = self.tokenized_molecules[idx]["attention_mask"][0]
        return sample_dict

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def create_dataset_weights(dataset, positive_class_weight=-1.0):
    count_dict = {}
    for item in dataset:
        label = item["labels"]
        if count_dict.get(label) is None:
            count_dict[label] = 0
        count_dict[label] += 1
    num_samples = len(dataset)
    label_to_weight = {}
    assert num_samples == sum(count_dict.values())
    count_0 = count_dict[0]
    count_1 = count_dict[1]
    freq_0 = count_0 / num_samples
    freq_1 = count_1 / num_samples
    label_to_weight[0] = 1 - freq_0
    if positive_class_weight <= 0:
        label_to_weight[1] = 1 - freq_1
    else:
        label_to_weight[1] = label_to_weight[0] * positive_class_weight
    sample_weights = np.empty(num_samples, dtype=np.float)
    for i, item in enumerate(dataset):
        label = item["labels"]
        sample_weights[i] = label_to_weight[label]
    return sample_weights


def train(model, iterator, optimizer, criterion, use_drug_embeddings=True, cross_att_flag=False,
          drug_features_size=None):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        drug_features = None
        if drug_features_size is not None:
            drug_features = batch["drug_features"].to(device)
        assert not (cross_att_flag and use_drug_embeddings)
        if use_drug_embeddings:
            drug_embeddings = batch["drug_embeddings"].to(device)
            output = model(inputs=input_ids, attention_mask=attention_mask, drug_embeddings=drug_embeddings,
                           drug_features=drug_features).squeeze(1)
        elif cross_att_flag:
            molecule_input_ids = batch["molecule_input_ids"].to(device)
            molecule_attention_mask = batch["molecule_attention_mask"].to(device)
            output = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                           molecule_inputs=molecule_input_ids, drug_features=drug_features,
                           molecule_attention_mask=molecule_attention_mask).squeeze(1)
        else:
            output = model(inputs=input_ids, attention_mask=attention_mask, drug_features=drug_features).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())

    return epoch_loss / (i + 1)


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


def evaluate(model, iterator, criterion, use_drug_embeddings, cross_att_flag=False, drug_features_size=None):
    model.eval()
    epoch_loss = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            drug_features = None
            if drug_features_size is not None:
                drug_features = batch["drug_features"].to(device)

            true_labels.extend(labels.cpu().numpy())
            labels = labels.to(device)

            assert not (cross_att_flag and use_drug_embeddings)
            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                output = model(inputs=input_ids, attention_mask=attention_mask,
                               drug_embeddings=drug_embeddings, drug_features=drug_features).squeeze(1)
            elif cross_att_flag:
                molecule_input_ids = batch["molecule_input_ids"].to(device)
                molecule_attention_mask = batch["molecule_attention_mask"].to(device)
                output = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                               molecule_inputs=molecule_input_ids, drug_features=drug_features,
                               molecule_attention_mask=molecule_attention_mask).squeeze(1)
            else:
                output = model(inputs=input_ids, attention_mask=attention_mask, drug_features=drug_features).squeeze(1)
            pred_probas = output.cpu().numpy()
            batch_pred_labels = (pred_probas >= 0.5) * 1

            loss = criterion(output, labels)

            pred_labels.extend(batch_pred_labels)
            epoch_loss += loss.item()

    valid_f1_score = f1_score(true_labels, pred_labels)
    return epoch_loss / (i + 1), valid_f1_score


def train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, n_epochs, use_drug_embeddings,
                   save_checkpoint_path, output_evaluation_path, cross_att_flag=False, drug_features_size=None):
    train_history = []
    valid_history = []
    valid_history_f1 = []

    best_valid_loss = float('inf')
    best_f1_score = 0.0
    best_epoch = -1

    eval_dir = os.path.dirname(output_evaluation_path)
    train_statistics_path = os.path.join(eval_dir, "training_logs.txt")

    for epoch in tqdm(range(n_epochs)):

        start_time = time.time()

        train_loss = train(bert_classifier, train_loader, optimizer, criterion, use_drug_embeddings,
                           cross_att_flag=cross_att_flag, drug_features_size=drug_features_size)
        valid_loss, valid_f1_score = evaluate(bert_classifier, dev_loader, criterion, use_drug_embeddings,
                                              cross_att_flag=cross_att_flag, drug_features_size=drug_features_size)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        valid_history_f1.append(valid_f1_score)

        if valid_f1_score > best_f1_score:
            best_f1_score = valid_f1_score
            best_epoch = epoch
            torch.save(bert_classifier.state_dict(), save_checkpoint_path)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}')

        with codecs.open(train_statistics_path, 'a+', encoding="utf-8") as output_path:
            output_path.write(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
            output_path.write(f'\tTrain Loss: {train_loss:.3f}\n')
            output_path.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}\n')

    return best_epoch


def train_evaluate_model(seed, bert_classifier, use_drug_embeddings, criterion, learning_rate, train_loader, dev_loader,
                         test_loader, num_epochs, output_evaluation_path, output_model_dir, model_chkpnt_name,
                         cross_att_flag=False, drug_features_size=None):
    torch.manual_seed(seed)
    optimizer = optim.Adam(bert_classifier.parameters(), lr=learning_rate)
    # criterion = nn.BCEWithLogitsLoss()

    output_ckpt_path = os.path.join(output_model_dir, f"best-val-{model_chkpnt_name}.pt")
    best_epoch = train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, num_epochs,
                                use_drug_embeddings, output_ckpt_path, output_evaluation_path,
                                cross_att_flag=cross_att_flag, drug_features_size=drug_features_size)
    if best_epoch != -1:
        bert_classifier.load_state_dict(torch.load(output_ckpt_path))

    true_labels, pred_labels, pred_probas = predict(bert_classifier, train_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag,
                                                    drug_features_size=drug_features_size)
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_train_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_train_probas.txt"), labels=pred_labels,
                       probas=pred_probas)
    true_labels, pred_labels, pred_probas = predict(bert_classifier, dev_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag,
                                                    drug_features_size=drug_features_size)
    assert len(pred_labels) == len(pred_probas)
    assert len(true_labels) == len(pred_labels)
    val_model_precision = precision_score(true_labels, pred_labels)
    val_model_recall = recall_score(true_labels, pred_labels)
    val_model_f1 = f1_score(true_labels, pred_labels)
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_dev_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_dev_probas.txt"), labels=pred_labels,
                       probas=pred_probas)

    true_labels, pred_labels, pred_probas = predict(bert_classifier, test_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag,
                                                    drug_features_size=drug_features_size)
    assert len(pred_labels) == len(pred_probas)
    assert len(true_labels) == len(pred_labels)
    test_model_precision = precision_score(true_labels, pred_labels)
    test_model_recall = recall_score(true_labels, pred_labels)
    test_model_f1 = f1_score(true_labels, pred_labels)
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_test_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_test_probas.txt"), labels=pred_labels,
                       probas=pred_probas)

    with codecs.open(output_evaluation_path, 'a+', encoding="utf-8") as output_file:
        output_file.write(f"{model_chkpnt_name},{best_epoch},{val_model_precision},{val_model_recall},{val_model_f1}\n")
        output_file.write(
            f"{model_chkpnt_name},{best_epoch},{test_model_precision},{test_model_recall},{test_model_f1}\n")

    del optimizer
    del criterion


def predict(model, data_loader, use_drug_embeddings, cross_att_flag=False, decision_threshold=0.5,
            drug_features_size=None):
    true_labels = []
    pred_labels = []
    pred_probas = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_true_labels = batch["labels"].cpu().numpy()
            drug_features = None
            if drug_features_size is not None:
                drug_features = batch["drug_features"].to(device)
            assert not (cross_att_flag and use_drug_embeddings)
            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                batch_pred_probas = model(inputs=input_ids, attention_mask=attention_mask,
                                          drug_embeddings=drug_embeddings, drug_features=drug_features).squeeze(1)
            elif cross_att_flag:
                molecule_input_ids = batch["molecule_input_ids"].to(device)
                molecule_attention_mask = batch["molecule_attention_mask"].to(device)
                batch_pred_probas = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                                          molecule_inputs=molecule_input_ids, drug_features=drug_features,
                                          molecule_attention_mask=molecule_attention_mask).squeeze(1)
            else:
                batch_pred_probas = model(inputs=input_ids, attention_mask=attention_mask,
                                          drug_features=drug_features, ).squeeze(1)

            batch_pred_probas = batch_pred_probas.cpu().numpy()

            batch_pred_labels = (batch_pred_probas >= decision_threshold) * 1

            pred_labels.extend(batch_pred_labels)
            true_labels.extend(batch_true_labels)
            pred_probas.extend(batch_pred_probas)
    return true_labels, pred_labels, pred_probas


class BertSimpleClassifier(nn.Module):
    def __init__(self, bert_text_encoder, dropout, drug_features_size):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        bert_hidden_dim = bert_text_encoder.config.hidden_size
        self.drug_features_size = drug_features_size
        self.emb_dropout = nn.Dropout(p=dropout)
        classifier_input_size = bert_hidden_dim
        if drug_features_size is not None:
            classifier_input_size += drug_features_size
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(classifier_input_size, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, drug_features=None, ):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        text_cls_embeddings = self.emb_dropout(text_cls_embeddings)
        if self.drug_features_size is not None:
            text_cls_embeddings = torch.cat([text_cls_embeddings, drug_features], dim=1)
        proba = self.classifier(text_cls_embeddings)
        return proba


class BertClassifierWithDrugEmbeddings(nn.Module):
    def __init__(self, bert_text_encoder, drug_enc_hid_dim, dropout, drug_features_size=None):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        # self.dropout = nn.Dropout(dropout)
        bert_hidden_dim = bert_text_encoder.config.hidden_size
        self.drug_features_size = drug_features_size
        self.emb_dropout = nn.Dropout(p=dropout)
        classifier_input_size = bert_hidden_dim + drug_enc_hid_dim
        if drug_features_size is not None:
            classifier_input_size += drug_features_size

        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(classifier_input_size, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, drug_embeddings, drug_features):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        text_cls_embeddings = self.emb_dropout(text_cls_embeddings)
        if self.drug_features_size is None:
            concat_text_drug_embeddings = torch.cat([text_cls_embeddings, drug_embeddings], dim=1)
        else:
            concat_text_drug_embeddings = torch.cat([text_cls_embeddings, drug_embeddings, drug_features], dim=1)

        proba = self.classifier(concat_text_drug_embeddings)
        return proba


class ConcatDoubleEncoderBertClassifieer(nn.Module):
    def __init__(self, bert_text_encoder, bert_molecule_encoder, classifier_dropout, ):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        self.bert_molecule_encoder = bert_molecule_encoder
        text_bert_hidden_dim = bert_text_encoder.config.hidden_size
        molecule_bert_hidden_dim = bert_molecule_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim + molecule_bert_hidden_dim, text_bert_hidden_dim),
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim, 1),
        )

    def forward(self, text_inputs, text_attention_mask, molecule_inputs, molecule_attention_mask, ):
        text_last_hidden_states = self.bert_text_encoder(text_inputs, attention_mask=text_attention_mask,
                                                         return_dict=True)['last_hidden_state']

        molecule_last_hidden_states = \
            self.bert_molecule_encoder(molecule_inputs, attention_mask=molecule_attention_mask,
                                       return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in text_last_hidden_states])
        molecule_cls_embeddings = torch.stack([elem[0, :] for elem in molecule_last_hidden_states])
        concat_text_drug_embeddings = torch.cat([text_cls_embeddings, molecule_cls_embeddings], dim=1)

        proba = self.classifier(concat_text_drug_embeddings)
        return proba


class CrossModalityBertClassifier(nn.Module):
    def __init__(self, bert_text_encoder, bert_molecule_encoder, classifier_dropout, cross_att_attention_dropout,
                 cross_att_hidden_dropout, drug_features_size=None):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        self.bert_molecule_encoder = bert_molecule_encoder
        self.drug_features_size = drug_features_size
        text_bert_hidden_dim = bert_text_encoder.config.hidden_size
        molecule_bert_hidden_dim = bert_molecule_encoder.config.hidden_size
        num_attention_heads = text_bert_hidden_dim // 64
        self.cross_attention_layer = BertCrossattLayer(text_bert_hidden_dim, molecule_bert_hidden_dim,
                                                       cross_att_attention_dropout, cross_att_hidden_dropout,
                                                       num_attention_heads=num_attention_heads)
        classifier_input_size = text_bert_hidden_dim
        if drug_features_size is not None:
            classifier_input_size += drug_features_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(classifier_input_size, text_bert_hidden_dim),
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim, 1),
        )

    def forward(self, text_inputs, text_attention_mask, molecule_inputs, molecule_attention_mask, drug_features):
        text_last_hidden_states = self.bert_text_encoder(text_inputs, attention_mask=text_attention_mask,
                                                         return_dict=True)['last_hidden_state']
        # text_cls_embeddings = torch.stack([elem[0, :] for elem in text_last_hidden_states])
        molecule_last_hidden_states = \
            self.bert_molecule_encoder(molecule_inputs, attention_mask=molecule_attention_mask,
                                       return_dict=True)['last_hidden_state']
        # molecule_cls_embeddings = torch.stack([elem[0, :] for elem in molecule_last_hidden_states])
        cross_attention_output = self.cross_attention_layer(input_tensor=text_last_hidden_states,
                                                            ctx_tensor=molecule_last_hidden_states, )
        cross_att_output_cls_embs = torch.stack([elem[0, :] for elem in cross_attention_output])
        if self.drug_features_size is not None:
            cross_att_output_cls_embs = torch.cat([cross_att_output_cls_embs, drug_features], dim=1)
        proba = self.classifier(cross_att_output_cls_embs)
        return proba


class DrugWithAttentionBertClassifier(nn.Module):
    def __init__(self, bert_text_encoder, drug_enc_hid_dim, cross_att_attention_dropout,
                 cross_att_hidden_dropout, classifier_dropout):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        text_bert_hidden_dim = bert_text_encoder.config.hidden_size
        num_attention_heads = text_bert_hidden_dim // 64
        self.cross_attention_layer = BertCrossattLayer(text_bert_hidden_dim, drug_enc_hid_dim,
                                                       cross_att_attention_dropout, cross_att_hidden_dropout,
                                                       num_attention_heads=num_attention_heads)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim, text_bert_hidden_dim),
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, drug_embeddings):
        test_last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                         return_dict=True)['last_hidden_state']
        cross_attention_output = self.cross_attention_layer(input_tensor=test_last_hidden_states,
                                                            ctx_tensor=drug_embeddings, )
        cross_att_output_cls_embs = torch.stack([elem[0, :] for elem in cross_attention_output])
        proba = self.classifier(cross_att_output_cls_embs)

        return proba


def clear():
    os.system('cls')


def get_positive_class_loss_weight(data_df, class_column="class"):
    class_counts = data_df[class_column].value_counts()
    positive_class_weight = class_counts[0] / class_counts[1]
    return positive_class_weight


def get_row_sider_embedding(row):
    embedding = row.loc["0":"1319"].astype(np.float).values
    return embedding


def main():
    config = configparser.ConfigParser()
    config.read("tune_config_14.ini")
    drug_embeddings_from = config["INPUT"]["DRUG_EMBEDDINGS_FROM"]
    max_length = config.getint("PARAMETERS", "MAX_TEXT_LENGTH")
    max_chemberta_length = config.getint("PARAMETERS", "MAX_MOLECULE_LENGTH")
    batch_size = config.getint("PARAMETERS", "BATCH_SIZE")
    learning_rate = config.getfloat("PARAMETERS", "LEARNING_RATE")
    dropout_p = config.getfloat("PARAMETERS", "DROPOUT")
    num_epochs = config.getint("PARAMETERS", "NUM_EPOCHS")
    text_encoder_name = config.get("PARAMETERS", "TEXT_ENCODER_NAME")
    apply_upsampling = config.getboolean("PARAMETERS", "APPLY_UPSAMPLING")
    freeze_layer_count = config.getint("PARAMETERS", "FREEZE_LAYER_COUNT")
    freeze_embeddings_layer = config.getboolean("PARAMETERS", "FREEZE_EMBEDDINGS_LAYER")
    use_weighted_loss = config.getboolean("PARAMETERS", "USE_WEIGHTED_LOSS")
    loss_weight = config.getfloat("PARAMETERS", "LOSS_WEIGHT")
    model_type = config["PARAMETERS"]["MODEL_TYPE"]
    mask_drug_flag = config.get("PARAMETERS", "MASK_DRUG")
    train_drug_sampling_type = config["PARAMETERS"]["DRUG_SAMPLING"]
    drug_features_path = config["PARAMETERS"]["DRUG_FEATURES_PATH"]
    if drug_features_path != "none":
        drug_features_dict = load_drug_features(drug_features_path)
        drug_features_fname = os.path.basename(drug_features_path)
        drug_features_str = drug_features_fname.split('.')[0]
        drug_features_size = len(list(drug_features_dict.values())[0])
    else:
        drug_features_dict = None
        drug_features_size = None
        drug_features_str = ''

    output_dir = config["OUTPUT"]["OUTPUT_DIR"]
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_evaluation_filename = config["OUTPUT"]["EVALUATION_FILENAME"]
    output_evaluation_path = os.path.join(output_dir, output_evaluation_filename)
    if apply_upsampling and use_weighted_loss:
        raise AssertionError(f"You can use only either weighted loss or upsampling")

    seed = 42
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    data_dir = config["INPUT"]["INPUT_DIR"]
    exp_description = f"{drug_features_str}"

    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")
    train_df = pd.read_csv(train_path, sep='\t', )
    dev_df = pd.read_csv(dev_path, sep='\t', )
    test_df = pd.read_csv(test_path, sep='\t', )

    drugs_dictionary = None
    if mask_drug_flag:
        exp_description += "_masking"
        drugs_dict_path = config["PARAMETERS"]["DRUG_DICT_PATH"]
        drugs_dictionary = load_drugs_dict(drugs_dict_path)

    if drug_embeddings_from == "chemberta":
        chemberta_model = RobertaModel.from_pretrained("./models/seyonec/ChemBERTa_zinc250k_v2_40k/model", ).to(
            device)
        tokenizer = AutoTokenizer.from_pretrained("./models/seyonec/ChemBERTa_zinc250k_v2_40k/model", )
        train_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                                   smiles_list=train_df.smiles.values,
                                                   max_length=max_chemberta_length, )
        dev_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                                 smiles_list=dev_df.smiles.values, max_length=max_chemberta_length, )
        test_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                                  smiles_list=test_df.smiles.values, max_length=max_chemberta_length, )
        chemberta_model = chemberta_model.cpu()
        del chemberta_model
    else:
        raise ValueError(f"Invalid drug embeddings source: {drug_embeddings_from}")
    print(
        f"Datasets sizes: mono_train {train_df.shape[0]},\n"
        f"dev: {dev_df.shape[0]},\n"
        f"test: {test_df.shape[0]}")

    if use_weighted_loss:
        if loss_weight < 0:
            pos_weight = [get_positive_class_loss_weight(train_df, ), ]
        else:
            pos_weight = [loss_weight, ]
        print("pos_weight", pos_weight)
        exp_description += f"_weighted_loss_{pos_weight}"
        pos_weight = torch.FloatTensor(pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(f"./models/{text_encoder_name}/model", )
    chemberta_tokenizer = None

    train_tweets_dataset = TweetsDataset(train_df, text_tokenizer, text_max_length=max_length,
                                         drugs_dictionary=drugs_dictionary, drug_features_dict=drug_features_dict,
                                         molecule_tokenizer=chemberta_tokenizer, drug_features_size=drug_features_size,
                                         sampling_type=train_drug_sampling_type)
    dev_tweets_dataset = TweetsDataset(dev_df, text_tokenizer, text_max_length=max_length,
                                       drug_features_dict=drug_features_dict, drug_features_size=drug_features_size,
                                       drugs_dictionary=drugs_dictionary, molecule_tokenizer=chemberta_tokenizer, )
    test_tweets_dataset = TweetsDataset(test_df, text_tokenizer, text_max_length=max_length,
                                        drug_features_dict=drug_features_dict, drug_features_size=drug_features_size,
                                        drugs_dictionary=drugs_dictionary, molecule_tokenizer=chemberta_tokenizer, )
    if apply_upsampling:

        positive_class_weight = config.getfloat("UPSAMPLING", "UPSAMPLING_WEIGHT")
        exp_description += f"_upsampling_{positive_class_weight}"
        train_weights = create_dataset_weights(train_tweets_dataset, positive_class_weight)
        print("Sampling weights:", set(train_weights))
        train_weights = torch.DoubleTensor(train_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        shuffle = False
    else:
        positive_class_weight = 0.0
        sampler = None
        shuffle = True

    seeds_list = [0, 1, 2, 3, 5, 7, 11, 13, 21, 42]

    setup_path = os.path.join(output_dir,
                              f"exp_{freeze_embeddings_layer}_{freeze_layer_count}{exp_description}/setup_descr.txt")
    setup_dir = os.path.dirname(setup_path)
    if not os.path.exists(setup_dir) and setup_dir != '':
        os.makedirs(setup_dir)
    write_hyperparams(apply_upsampling, positive_class_weight, num_epochs, dropout_p, freeze_layer_count,
                      freeze_embeddings_layer, text_encoder_name, setup_path)
    for seed in seeds_list:
        experiment_dir = f"exp_{freeze_embeddings_layer}_{freeze_layer_count}{exp_description}/seed_{seed}"
        experiment_dir = os.path.join(output_dir, experiment_dir)
        if not os.path.exists(experiment_dir) and experiment_dir != '':
            os.makedirs(experiment_dir)

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.random.manual_seed(seed)
        torch.cuda.random.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        bert_text_encoder = AutoModel.from_pretrained(f"./models/{text_encoder_name}/model", )

        if freeze_layer_count > 0:
            for layer in bert_text_encoder.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

        if freeze_embeddings_layer:
            for param in bert_text_encoder.embeddings.parameters():
                param.requires_grad = False
        print("#Trainable params: ", sum(p.numel() for p in bert_text_encoder.parameters() if p.requires_grad))

        num_workers = 4

        train_loader = torch.utils.data.DataLoader(
            train_tweets_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=shuffle,
            drop_last=True,
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
        )
        cross_att_flag = False

        torch.manual_seed(seed)
        use_drug_embeddings = False
        bert_classifier = BertSimpleClassifier(bert_text_encoder, dropout=dropout_p,
                                               drug_features_size=drug_features_size).to(device)
        checkpoint_name = f"simple_{text_encoder_name.split('/')[-1]}"
        model_save_dir = os.path.join(output_dir,
                                      f"exp_{freeze_embeddings_layer}_{freeze_layer_count}{exp_description}/")
        train_evaluate_model(seed, bert_classifier, use_drug_embeddings, criterion, learning_rate, train_loader,
                             dev_loader, test_loader, num_epochs, output_evaluation_path, model_save_dir,
                             checkpoint_name,
                             cross_att_flag=cross_att_flag, drug_features_size=drug_features_size)

        true_labels, dev_pred_labels, dev_pred_probas = predict(bert_classifier, dev_loader, use_drug_embeddings,
                                                                drug_features_size=drug_features_size)
        assert len(dev_pred_labels) == len(true_labels)
        assert len(dev_pred_labels) == len(dev_pred_probas)
        dev_precision = precision_score(true_labels, dev_pred_labels)
        dev_recall = recall_score(true_labels, dev_pred_labels)
        dev_f1 = f1_score(true_labels, dev_pred_labels)

        print(f"{dev_precision},{dev_recall},{dev_f1}")

        true_labels, test_pred_labels, test_pred_probas = predict(bert_classifier, test_loader, use_drug_embeddings,
                                                                  drug_features_size=drug_features_size)
        assert len(test_pred_labels) == len(true_labels)
        assert len(test_pred_labels) == len(test_pred_probas)
        test_precision = precision_score(true_labels, test_pred_labels)
        test_recall = recall_score(true_labels, test_pred_labels)
        test_f1 = f1_score(true_labels, test_pred_labels)
        print(f"{test_precision},{test_recall},{test_f1}")

        exp_scores_path = os.path.join(experiment_dir, "scores.txt")
        with codecs.open(exp_scores_path, 'a+', encoding="utf-8") as out_file:
            out_file.write(
                f"{seed},{dev_precision},{dev_recall},{dev_f1},{test_precision},{test_recall},{test_f1}\n")

        dev_labels_path = os.path.join(experiment_dir, "dev_labels.txt")
        dev_probas_path = os.path.join(experiment_dir, "dev_probas.txt")
        test_labels_path = os.path.join(experiment_dir, "test_labels.txt")
        test_probas_path = os.path.join(experiment_dir, "test_probas.txt")

        save_labels_probas(dev_labels_path, dev_probas_path, dev_pred_labels, dev_pred_probas)
        save_labels_probas(test_labels_path, test_probas_path, test_pred_labels, test_pred_probas)

        bert_classifier = bert_classifier.cpu()
        bert_text_encoder = bert_text_encoder.cpu()
        del bert_classifier
        del bert_text_encoder


if __name__ == '__main__':
    main()
