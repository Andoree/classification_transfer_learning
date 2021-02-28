import codecs
import configparser
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

LABELS = ["EF", "INF", "ADR", "DI", "Finding"]


class SentencesDataset(Dataset):
    def __init__(self, data_df, text_tokenizer, max_length=128, text_column="sentences"):
        self.labels = data_df[LABELS].astype(np.float32).values
        self.max_length = max_length
        self.tokenized_tweets = [text_tokenizer.encode_plus(x, max_length=self.max_length,
                                                            padding="max_length", truncation=True,
                                                            return_tensors="pt", ) for x in data_df[text_column].values]

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_tweets[idx]["input_ids"][0],
            "attention_mask": self.tokenized_tweets[idx]["attention_mask"][0],
            "labels": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def train_multilabel(model, iterator, optimizer, criterion, device):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(inputs=input_ids, attention_mask=attention_mask, )
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())

    return epoch_loss / (i + 1)


def evaluate_multilabel(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            true_labels.extend(labels.cpu().numpy())
            labels = labels.to(device)

            output = model(inputs=input_ids, attention_mask=attention_mask, )

            pred_probas = output.cpu().numpy()
            batch_pred_labels = (pred_probas >= 0.5) * 1

            loss = criterion(output, labels)

            pred_labels.extend(batch_pred_labels)
            epoch_loss += loss.item()

    valid_f1_score = f1_score(true_labels, pred_labels, average='macro')
    return epoch_loss / (i + 1), valid_f1_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_evaluate_multilabel(bert_classifier, train_loader, dev_loader, optimizer, criterion, n_epochs,
                              checkpoint_fname, device, output_model_dir, ):
    train_history = []
    valid_history = []
    valid_history_f1 = []

    best_f1_score = 0.0

    for epoch in tqdm(range(n_epochs)):

        start_time = time.time()

        train_loss = train_multilabel(bert_classifier, train_loader, optimizer, criterion, device)
        valid_loss, valid_f1_score = evaluate_multilabel(bert_classifier, dev_loader, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        valid_history_f1.append(valid_f1_score)

        if valid_f1_score > best_f1_score:
            best_f1_score = valid_f1_score
            torch.save(bert_classifier.state_dict(), os.path.join(output_model_dir, f'best-val-{checkpoint_fname}.pt'))

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}')


def predict_multilabel(model, data_loader, device):
    true_labels = []
    pred_labels = []
    pred_probas = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_true_labels = batch["labels"].cpu().numpy()

            batch_pred_probas = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)

            batch_pred_probas = batch_pred_probas.cpu().numpy()
            pred_probas.extend(batch_pred_probas)

            batch_pred_labels = (batch_pred_probas >= 0.5) * 1

            pred_labels.extend(batch_pred_labels)
            true_labels.extend(batch_true_labels)
    return true_labels, pred_labels, pred_probas


class BertMultilabelClassifier(nn.Module):
    def __init__(self, bert_text_encoder, dropout=0.3):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        bert_hidden_dim = bert_text_encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, len(LABELS)),
        )

    def forward(self, inputs, attention_mask, ):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])

        proba = self.classifier(text_cls_embeddings)
        return proba


def save_multilabel_labels_probas(labels_path, probas_path, labels, probas):
    with codecs.open(labels_path, 'w+', encoding="utf-8") as labels_file, \
            codecs.open(probas_path, 'w+', encoding="utf-8") as probas_file:
        for label, probabilities in zip(labels, probas):
            labels_file.write(f"{label}\n")
            probas_file.write(f"{','.join(probabilities)}\n")


def main():
    config = configparser.ConfigParser()
    config.read("config_pretrain.ini")
    data_dir = config["INPUT"]["INPUT_DIR"]
    tweets_dir = config["INPUT"]["TWEETS_DIR"]
    seed = config.getint("PARAMETERS", "SEED")
    max_seq_length = config.getint("PARAMETERS", "MAX_SEQ_LENGTH")
    batch_size = config.getint("PARAMETERS", "BATCH_SIZE")
    learning_rate = config.getfloat("PARAMETERS", "LEARNING_RATE")
    dropout_p = config.getfloat("PARAMETERS", "DROPOUT")
    num_epochs = config.getint("PARAMETERS", "NUM_EPOCHS")
    model_name = config.get("PARAMETERS", "MODEL_NAME")
    output_checkpoint_name = config["OUTPUT"]["CHKPT_NAME"]
    output_dir = config["OUTPUT"]["OUTPUT_DIR"]
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_evaluation_filename = config["OUTPUT"]["EVALUATION_FILENAME"]
    output_eval_path = os.path.join(output_dir, output_evaluation_filename)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available else "cpu"

    train_path = os.path.join(data_dir, "train.csv")
    dev_path = os.path.join(data_dir, "dev.csv")
    tweets_test_path = os.path.join(tweets_dir, "test.tsv")
    tweets_dev_path = os.path.join(tweets_dir, "dev.tsv")
    tweets_dev_df = pd.read_csv(tweets_dev_path, sep='\t', quoting=3)
    tweets_test_df = pd.read_csv(tweets_test_path, sep='\t', quoting=3)
    for label in LABELS:
        tweets_dev_df[label] = 0
        tweets_test_df[label] = 0

    train_df = pd.read_csv(train_path, encoding="utf-8")
    dev_df = pd.read_csv(dev_path, encoding="utf-8")

    text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretrained_bert = AutoModel.from_pretrained(model_name)

    train_dataset = SentencesDataset(train_df, text_tokenizer, max_length=max_seq_length)
    dev_dataset = SentencesDataset(dev_df, text_tokenizer, max_length=max_seq_length)
    tweets_dev_dataset = SentencesDataset(tweets_dev_df, text_tokenizer, max_length=max_seq_length, text_column="tweet")
    tweets_test_dataset = SentencesDataset(tweets_test_df, text_tokenizer, max_length=max_seq_length,
                                           text_column="tweet")

    num_workers = 4

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    tweets_dev_loader = torch.utils.data.DataLoader(
        tweets_dev_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    tweets_test_loader = torch.utils.data.DataLoader(
        tweets_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )

    torch.manual_seed(seed)

    bert_clf = BertMultilabelClassifier(pretrained_bert, dropout=dropout_p).to(device)
    optimizer = optim.Adam(bert_clf.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_evaluate_multilabel(bert_clf, train_loader, dev_loader, optimizer, criterion, num_epochs,
                              output_checkpoint_name, device, output_dir)

    bert_clf.load_state_dict(torch.load(os.path.join(output_dir, f'best-val-{output_checkpoint_name}.pt')))

    true_labels, pred_labels, pred_probas = predict_multilabel(bert_clf, dev_loader, device)
    tweets_dev_true_labels, tweets_dev_pred_labels, tweets_dev_pred_probas \
        = predict_multilabel(bert_clf, tweets_dev_loader, device)
    tweets_test_true_labels, tweets_test_pred_labels, tweets_test_pred_probas = predict_multilabel(bert_clf,
                                                                                                   tweets_test_loader,
                                                                                                   device)

    save_multilabel_labels_probas(labels_path=os.path.join(output_dir, "pred_dev_labels.txt"),
                                  probas_path=os.path.join(output_dir, "pred_dev_probas.txt"),
                                  labels=tweets_dev_pred_labels,
                                  probas=tweets_dev_pred_probas)
    save_multilabel_labels_probas(labels_path=os.path.join(output_dir, "pred_test_labels.txt"),
                                  probas_path=os.path.join(output_dir, "pred_test_probas.txt"),
                                  labels=tweets_test_pred_labels,
                                  probas=tweets_test_pred_probas)

    with codecs.open(output_eval_path, 'w+', encoding="utf-8") as output_file:
        output_file.write(f"train size: {len(train_loader)}\n")
        output_file.write(f"Dev size: {len(dev_loader)}\n")
        output_file.write(f"P: {precision_score(true_labels, pred_labels,  average='macro')}\n")
        output_file.write(f"R: {recall_score(true_labels, pred_labels, average='macro')}\n")
        output_file.write(f"F1: {f1_score(true_labels, pred_labels, average='macro')}\n")
        output_file.write(f"F1: {classification_report(true_labels, pred_labels, )}\n")

    torch.save(bert_clf.bert_text_encoder.state_dict(), os.path.join(output_dir, f'{output_checkpoint_name}.pt'))


if __name__ == '__main__':
    main()
