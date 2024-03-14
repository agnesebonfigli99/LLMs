
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from datasets import load_dataset 
from sklearn import train_test_split, classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = load_dataset("tner/bionlp2004")
train_df = pd.DataFrame({'tokens': dataset['train']['tokens'], 'tags': dataset['train']['tags']})
validation_df = pd.DataFrame({'tokens': dataset['validation']['tokens'], 'tags': dataset['validation']['tags']})
test_df = pd.DataFrame({'tokens': dataset['test']['tokens'], 'tags': dataset['test']['tags']})

original_label_mapping = {
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell_type": 5,
    "I-cell_type": 6,
    "B-cell_line": 7,
    "I-cell_line": 8,
    "B-RNA": 9,
    "I-RNA": 10
}

updated_label_mapping = {
    0: 0,  
    1: 1,  
    2: 1,  
    3: 2,  
    4: 2,  
    5: 3,  
    6: 3,  
    7: 4,  
    8: 4,  
    9: 5,  
    10: 5  
}

def update_tags(tags_list, label_mapping):
    return [label_mapping[tag] if tag in label_mapping else tag for tag in tags_list]

train_df['tags'] = train_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))
validation_df['tags'] = validation_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))
test_df['tags'] = test_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data.tokens[index]
        word_labels = self.data.tags[index]
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        labels = [label_mapping[label] for label in word_labels]
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        return item

    def __len__(self):
        return self.len

def train(epoch, training_loader, model, optimizer):
    model.train()
    for _, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs[0]
        if _ % 5000 == 0:
            print(f"Loss after {_} steps: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def valid(model, testing_loader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            output = model(input_ids=ids, attention_mask=mask, labels=labels)
            eval_loss += output.loss.item()
            eval_logits = output.logits

            nb_eval_steps += 1

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # Flatten the outputs
            flattened_targets = labels.view(-1)
            active_logits = eval_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            active_accuracy = labels.view(-1) != -100

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    print(classification_report(labels, predictions))



MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-05 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium').eos_token
tokenizer = BertTokenizer.from_pretrained('mis-lab/biobert-v1.1')
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt").eos_token

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=6, output_attentions=True)
model = GPT2ForTokenClassification.from_pretrained('gpt2-medium', num_labels=6, output_attentions=True)
model= BertForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=6, output_attentions=True)
model = BioGptForTokenClassification.from_pretrained('microsoft/biogpt', num_labels=6, output_attentions=True)

sample_df = train_df.sample(frac=0.1, random_state=200)  
train_df, test_df, train_labels, _ = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=train_labels)  
training_set = CustomDataset(train_df, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch, training_loader, model, optimizer)

ids_to_labels = {0: "O", 1: "DNA", 2: "protein", 3: "cell_type", 4: "cell_line", 5: "RNA"}
labels, predictions = valid(model, testing_loader)
