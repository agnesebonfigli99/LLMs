# Required installations
# !pip install datasets
# !pip install transformers
# !pip install torch
# !pip install pytorch-crf
# !pip install sklearn
# !pip install tqdm

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (BertTokenizer, BertForSequenceClassification,
                          GPT2Tokenizer, GPT2ForSequenceClassification,
                          BioGptTokenizer, BioGptForSequenceClassification)
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences1[idx], self.sentences2[idx], self.labels[idx]

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def get_embeddings(model, tokenizer, data_loader, device):
    embeddings1 = []
    embeddings2 = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for sent1, sent2, _ in tqdm(data_loader):
            inputs1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True).to(device)
            inputs2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs1 = model(**inputs1, output_hidden_states=True).hidden_states[-1]
            outputs2 = model(**inputs2, output_hidden_states=True).hidden_states[-1]
            embeddings1.append(outputs1.mean(dim=1).cpu().numpy())
            embeddings2.append(outputs2.mean(dim=1).cpu().numpy())
    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)
    return embeddings1, embeddings2

def prepare_features(embeddings1, embeddings2):
    return [np.concatenate((emb1, emb2)) for emb1, emb2 in zip(embeddings1, embeddings2)]

def probing_task(train_features, train_labels, test_features, test_labels):
    clf = SVC()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    cm = confusion_matrix(test_labels, predictions)
    cr = classification_report(test_labels, predictions)
    return cm, cr

def main(model_name, training_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data('mli_train.jsonl')
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['gold_label'].map({'entailment': 0, 'contradiction': 1, 'neutral': 2}).values

    train_sentences1, test_sentences1, train_sentences2, test_sentences2, train_labels, test_labels = train_test_split(
        sentences1, sentences2, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_sentences1, train_sentences2, train_labels)
    test_dataset = CustomDataset(test_sentences1, test_sentences2, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model, tokenizer = load_model_and_tokenizer(model_name, training_size, device)

    folder_path = './embeddings'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_embeddings1, train_embeddings2 = get_embeddings(model, tokenizer, train_loader, device)
    test_embeddings1, test_embeddings2 = get_embeddings(model, tokenizer, test_loader, device)

    train_features = prepare_features(train_embeddings1, train_embeddings2)
    test_features = prepare_features(test_embeddings1, test_embeddings2)

    cm, cr = probing_task(train_features, train_labels, test_features, test_labels)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{cr}")

    np.save(os.path.join(folder_path, f'train_embeddings1.npy'), train_embeddings1)
    np.save(os.path.join(folder_path, f'train_embeddings2.npy'), train_embeddings2)

def load_model_and_tokenizer(model_name, training_size, device):
    model_map = {
        'bert': ('bert-base-cased', BertTokenizer, BertForSequenceClassification),
        'biobert': ('dmis-lab/biobert-v1.1', BertTokenizer, BertForSequenceClassification),
        'gpt2': ('gpt2-medium', GPT2Tokenizer, GPT2ForSequenceClassification),
        'biogpt': ('microsoft/biogpt', GPT2Tokenizer, GPT2ForSequenceClassification),  
    }

    model_path, tokenizer_class, model_class = model_map[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_path)

    if training_size == 0:
        model = model_class.from_pretrained(model_path, num_labels=3, output_attentions=True)
    else:
        path_to_finetuned_model_weights = f'/path/to/save/model_{model_name}_{training_size}.bin'
        model = model_class(num_labels=3, output_attentions=True)  
        state_dict = torch.load(path_to_finetuned_model_weights, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform a probing task with pre-trained/fine-tuned models.')
    parser.add_argument('--training_size', type=int, choices=[0, 10, 30, 50, 100], help='Size of the training set as a percentage')
    parser.add_argument('--model_name', type=str, choices=['bert', 'biobert', 'gpt2', 'biogpt'], help='Model to fine-tune')
    
    args = parser.parse_args()
    main(args.model_name, args.training_size)

