import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

# Ensure your CUDA devices are correctly configured if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
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

def get_embeddings(model, tokenizer, data_loader):
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

def main():
    # Load and preprocess the dataset
    data = load_data('mli_train.jsonl')
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['gold_label'].map({'entailment': 0, 'contradiction': 1, 'neutral': 2}).values

    # Split the dataset
    train_sentences1, test_sentences1, train_sentences2, test_sentences2, train_labels, test_labels = train_test_split(
        sentences1, sentences2, labels, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_sentences1, train_sentences2, train_labels)
    test_dataset = CustomDataset(test_sentences1, test_sentences2, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

    # Get embeddings for each layer
    folder_path = './embeddings'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for layer_num in range(model.config.num_hidden_layers + 1):  # Including the output layer
        print(f"Evaluating BERT layer {layer_num}")
        train_embeddings1, train_embeddings2 = get_embeddings(model, tokenizer, train_loader)
        test_embeddings1, test_embeddings2 = get_embeddings(model, tokenizer, test_loader)

        # Preparing features
        train_features = prepare_features(train_embeddings1, train_embeddings2)
        test_features = prepare_features(test_embeddings1, test_embeddings2)

        # Probing task
        cm, cr = probing_task(train_features, train_labels, test_features, test_labels)
        print(f"Layer {layer_num} - Confusion Matrix:\n{cm}")
        print(f"Layer {layer_num} - Classification Report:\n{cr}")

        # Save embeddings
        np.save(os.path.join(folder_path, f'train_embeddings1_layer_{layer_num}.npy'), train_embeddings1)
        np.save(os.path.join(folder_path, f'train_embeddings2_layer_{layer_num}.npy'), train_embeddings2)

if __name__ == "__main__":
    main()
