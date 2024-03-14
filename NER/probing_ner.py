# Required installations
# !pip install datasets
# !pip install transformers
# !pip install torch
# !pip install pytorch-crf
# !pip install sklearn
# !pip install tqdm

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification
from datasets import load_dataset
from torchcrf import CRF
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
dataset = load_dataset("tner/bionlp2004")

def dataset_to_dataframe(split):
    return pd.DataFrame({'tokens': dataset[split]['tokens'], 'tags': dataset[split]['tags']})
train_df = dataset_to_dataframe('train')
validation_df = dataset_to_dataframe('validation')
test_df = dataset_to_dataframe('test')

label_mapping = {
    "O": 0,
    "B-DNA": 1, "I-DNA": 1,
    "B-protein": 2, "I-protein": 2,
    "B-cell_type": 3, "I-cell_type": 3,
    "B-cell_line": 4, "I-cell_line": 4,
    "B-RNA": 5, "I-RNA": 5
}

def update_tags(tags_list, label_mapping):
    return [label_mapping[tag] if tag in label_mapping else tag for tag in tags_list]

for df in [train_df, validation_df, test_df]:
    df['tags'] = df['tags'].apply(lambda tags_list: update_tags(tags_list, label_mapping))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #GPT2Tokenizer.from_pretrained('gpt2-medium')
#tokenizer = BertTokenizer.from_pretrained('mis-lab/biobert-v1.1') or tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BertForTokenClassification.from_pretrained('bert-base-uncased') #GPT2ForTokenClassification.from_pretrained('gpt2-medium', num_labels=6)
#model = BertForTokenClassification.from_pretrained('mis-lab/biobert-v1.1', num_labels=6) or model = ForTokenClassification.from_pretrained("microsoft/biogpt", num_labels=6)
model.to(device)

def get_entity_embeddings_mean(model, tokenizer, sentences_tokens, tags, layer_num=-1):
    """
    Compute the mean embeddings for entities using BERT model.
    """
    model.eval() 
    entity_embeddings_means = []
    corresponding_labels = []
    corresponding_tokens = []

    with torch.no_grad():
        for tokens, tag_seq in zip(sentences_tokens, tags):
            sentence = " ".join(tokens)
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512, return_overflowing_tokens=True).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_num]

            entity_embedding_for_sent = []
            entity_labels_for_sent = []
            current_word_embedding = []
            corresponding_tokens_for_sent = []
            current_word_label = tag_seq[0]
            tokenized_sentence = tokenizer.tokenize(sentence)
    
            token_index = 0
            prev_token = tokenized_sentence[0]
            for token, token_embedding in zip(tokenized_sentence, hidden_states[0]):
                if token.startswith("##") : #Ġ
                    current_word_embedding.append(token_embedding.cpu().numpy())
                else:
                    if current_word_embedding:
                        mean_embedding = np.mean(current_word_embedding, axis=0)
                        entity_embedding_for_sent.append(mean_embedding)
                        entity_labels_for_sent.append(current_word_label)
                        corresponding_tokens_for_sent.append(prev_token)
                    current_word_embedding = [token_embedding.cpu().numpy()]
                    prev_token = token
                    if token_index < len(tag_seq): #token_index < len(tag_seq) - 1
                        current_word_label = tag_seq[token_index]
                        token_index += 1
            if current_word_embedding:
                mean_embedding = np.mean(current_word_embedding, axis=0)
                entity_embedding_for_sent.append(mean_embedding)
                entity_labels_for_sent.append(current_word_label)
                corresponding_tokens_for_sent.append(prev_token)
    
            entity_embeddings_means.append(np.array(entity_embedding_for_sent))
            corresponding_labels.append(np.array(entity_labels_for_sent))
            corresponding_tokens.append(corresponding_tokens_for_sent)
        return entity_embeddings_means, corresponding_labels, corresponding_tokens


class BiLSTMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_tags):
        super(BiLSTMCRF, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, labels=None):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        emissions = self.hidden2tag(lstm_out)
        if labels is not None:
            loss = -self.crf(emissions, labels, reduction='mean')
            return loss
        return self.crf.decode(emissions) 


def probing_task(entity_embeddings, train_labels, test_entity_embeddings, test_labels):
    """
    Train and evaluate a BiLSTM-CRF model on the provided embeddings.
    """
    train_features = torch.tensor(np.array(entity_embeddings), dtype=torch.float32)
    train_labels = torch.tensor(np.array(train_labels), dtype=torch.long)
    test_features = torch.tensor(np.array(test_entity_embeddings), dtype=torch.float32)
    test_labels = torch.tensor(np.array(test_labels), dtype=torch.long)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BiLSTMCRF(embedding_dim=768, hidden_dim=256, num_tags=6).to(device)
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(3):  
        for batch in train_loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model(features, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            features = features.to(device)
            batch_predictions = model(features)
            predictions.extend(batch_predictions)
            true_labels.extend(labels.tolist())

    flat_predictions = [p for sublist in predictions for p in sublist]
    flat_true_labels = [label for sublist in true_labels for label in sublist]

    f1_macro = f1_score(flat_true_labels, flat_predictions, average='macro')
    f1_micro = f1_score(flat_true_labels, flat_predictions, average='micro')

    return f1_macro, f1_micro

def main():
    base_path = "/path/to/your/embeddings/"  # Example base path

    bert_model, bert_tokenizer = None, None  
    train_tokens, train_tags, test_tokens, test_tags = None, None, None, None  
    layer_num = 0  

    train_entity_embeddings, train_filtered_tags, train_words = get_entity_embeddings_mean(bert_model, bert_tokenizer, train_tokens, train_tags, layer_num)
    test_entity_embeddings, test_filtered_tags, test_words = get_entity_embeddings_mean(bert_model, bert_tokenizer, test_tokens, test_tags, layer_num)

    train_entity_embeddings_filtered = [torch.tensor(embedding) for embedding in train_entity_embeddings]
    test_entity_embeddings_filtered = [torch.tensor(embedding) for embedding in test_entity_embeddings]

    results = {}
    for layer_num in range(0, 13):
        print(f"Probing layer {layer_num}...")
        f1_macro, f1_micro = probing_task(
            train_entity_embeddings_filtered,
            train_filtered_tags,
            test_entity_embeddings_filtered,
            test_filtered_tags,
            embedding_dim=768, 
            hidden_dim=256, 
            num_tags=6  
        )
        results[layer_num] = {
            "F1 Macro": f1_macro,
            "F1 Micro": f1_micro
        }

        print(f"Layer {layer_num}: F1 Macro = {f1_macro}, F1 Micro = {f1_micro}")
    print("Completed probing across all BERT layers.")

if __name__ == "__main__":
    main()
