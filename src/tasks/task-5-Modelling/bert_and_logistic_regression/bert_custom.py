import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import numpy as np

# NOTE: run data_cleaning.py first in order to generate twitter_hatespeech_dataset_after_cleaning.csv
df = pd.read_csv('../../../data/twitter_hatespeech_dataset_after_cleaning.csv')
pd.set_option('display.max_columns', None)
print(df.head())

df = df.iloc[:,[2,4]]
df_x = df.drop('label', axis='columns')
x_list = df_x['clean_tweet'].tolist()
y_list = list(df.label)

train_texts, val_texts, train_labels, val_labels = train_test_split(x_list, y_list, test_size=0.1)

MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_token_type_ids=False, add_special_tokens=True, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_token_type_ids=False, add_special_tokens=True, return_tensors='pt')

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        encoding_id = self.encodings.data['input_ids'][idx]
        encoding_mask = self.encodings.data['attention_mask'][idx]
        label = self.labels[idx]
        return {
            'input_ids': encoding_id.flatten(),
            'attention_mask': encoding_mask.flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# data loader
def get_data_loader(encodings, labels, batch_size=32):
  dataset = IMDbDataset(encodings, labels)

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
  )

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ClassifierNet(nn.Module):

    def __init__(self, n_classes):
        super(ClassifierNet, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    #note: https://stackoverflow.com/questions/65082243/dropout-argument-input-position-1-must-be-tensor-not-str-when-using-bert
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = ClassifierNet(n_classes=2)
model = model.to(device)

#training
epochs = 1
optim = AdamW(model.parameters(), lr=5e-5)
train_data_loader = get_data_loader(train_encodings, train_labels)
criterion = nn.CrossEntropyLoss().to(device)

best_accuracy = 0

for epoch in range(epochs):
    train_acc = 0
    train_loss = 0
    model = model.train()
    losses = []
    correct_predictions = 0
    for batch in train_data_loader:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        print("loss: ", loss.item())
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optim.step()

    train_acc = correct_predictions.double() / len(train_texts)
    train_loss = np.mean(losses)
    print("loss: ", train_loss)
    print("acc: ", train_acc)

print("END")