import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from torchcrf import CRF


def load_and_prepare_data():
    dataset = load_dataset("tner/bionlp2004")
    train_df = pd.DataFrame({'tokens': dataset['train']['tokens'], 'tags': dataset['train']['tags']})
    validation_df = pd.DataFrame({'tokens': dataset['validation']['tokens'], 'tags': dataset['validation']['tags']})
    test_df = pd.DataFrame({'tokens': dataset['test']['tokens'], 'tags': dataset['test']['tags']})
    return train_df, validation_df, test_df

def update_tags(tags_list, label_mapping):
    return [label_mapping[tag] if tag in label_mapping else tag for tag in tags_list]


def update_dataframe_tags(dataframes, label_mapping):
    for df in dataframes:
        df['tags'] = df['tags'].apply(lambda tags_list: update_tags(tags_list, label_mapping))


class BiLSTMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_tags):
        super(BiLSTMCRF, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, embeddings, labels=None, mask=None):
        lstm_out, _ = self.lstm(embeddings)
        logits = self.hidden2tag(lstm_out)
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(logits)


def get_entity_embeddings_mean_bert(model, tokenizer, sentences_tokens, tags, device, layer_num=-1):
    entity_embeddings_means = []
    corresponding_labels = []
    corresponding_tokens = []
    model.eval()

    for tokens, tag_seq in zip(sentences_tokens, tags):
        sentence = " ".join(tokens)
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_num]
        entity_embedding_for_sent, entity_labels_for_sent, corresponding_tokens_for_sent = process_hidden_states(hidden_states, tag_seq, tokenizer)
        entity_embeddings_means.append(np.array(entity_embedding_for_sent))
        corresponding_labels.append(np.array(entity_labels_for_sent))
        corresponding_tokens.append(corresponding_tokens_for_sent)
    return entity_embeddings_means, corresponding_labels, corresponding_tokens

def process_hidden_states(hidden_states, tag_seq, tokenizer):
    entity_embedding_for_sent = []
    entity_labels_for_sent = []
    current_word_embedding = []
    corresponding_tokens_for_sent = []
    current_word_label = tag_seq[0]
    tokenized_sentence = tokenizer.tokenize(sentence)
    token_index = 0
    prev_token = tokenized_sentence[0]
    for token, token_embedding in zip(tokenized_sentence, hidden_states[0]):
        if token.startswith("##"):
            current_word_embedding.append(token_embedding.cpu().numpy())
        else:
            if current_word_embedding:
                mean_embedding = np.mean(current_word_embedding, axis=0)
                entity_embedding_for_sent.append(mean_embedding)
                entity_labels_for_sent.append(current_word_label)
                corresponding_tokens_for_sent.append(prev_token)
            current_word_embedding = [token_embedding.cpu().numpy()]
            prev_token = token
            if token_index < len(tag_seq):
                current_word_label = tag_seq[token_index]
                token_index += 1
        if current_word_embedding:
            mean_embedding = np.mean(current_word_embedding, axis=0)
            entity_embedding_for_sent.append(mean_embedding)
            entity_labels_for_sent.append(current_word_label)
            corresponding_tokens_for_sent.append(prev_token)
    return entity_embedding_for_sent, entity_labels_for_sent, corresponding_tokens_for_sent


def probing_task(model, train_loader, test_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):  
        model.train()
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            mask = labels != -1  
            optimizer.zero_grad()
            loss = model(embeddings, labels, mask=mask)
            loss.backward()
            optimizer.step()
    
    
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            mask = labels != -1  
            predictions = model(embeddings, mask=mask)
            for pred, label, m in zip(predictions, labels, mask):
                all_predictions.extend([p for p, valid in zip(pred, m) if valid])
                all_labels.extend([l.item() for l, valid in zip(label, m) if valid])
    
    
    print_classification_report(all_labels, all_predictions)

def print_classification_report(true_labels, predictions):
    print("Classification Report:")
    print(classification_report(true_labels, predictions))
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_micro = f1_score(true_labels, predictions, average='micro')
    print(f"F1 Score (Macro): {f1_macro}\nF1 Score (Micro): {f1_micro}")



