from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# NOTE: run data_cleaning.py first in order to generate twitter_hatespeech_dataset_after_cleaning.csv
df = pd.read_csv('../../../data/twitter_hatespeech_dataset_after_cleaning.csv')
pd.set_option('display.max_columns', None)
print(df.head())
print("shape: ", df.shape)
print("len: ", len(df))

df = df.iloc[:,[2,4]]
df_x = df.drop('label', axis='columns')
x_list = df_x['clean_tweet'].tolist()
y_list = list(df.label)

train_texts, val_texts, train_labels, val_labels = train_test_split(x_list, y_list, test_size=0.1)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

# FINE_TUNING with TRAINER

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch"
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

import numpy as np
from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

print("END")