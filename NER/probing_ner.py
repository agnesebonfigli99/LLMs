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
from transformers import (BertTokenizer, BertForTokenClassification,
                          GPT2Tokenizer, GPT2ForTokenClassification,
                          BioGptTokenizer, BioGptForTokenClassification)
from datasets import load_dataset
from torchcrf import CRF
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_to_dataframe(split):
    dataset = load_dataset("tner/bionlp2004")
    return pd.DataFrame({'tokens': dataset[split]['tokens'], 'tags': dataset[split]['tags']})

def update_tags(tags_list, label_mapping):
    return [label_mapping[tag] if tag in label_mapping else tag for tag in tags_list]

for df in [train_df, validation_df, test_df]:
    df['tags'] = df['tags'].apply(lambda tags_list: update_tags(tags_list, label_mapping))

model_map = {
    'bert': ('bert-base-uncased', BertTokenizer, BertForSequenceClassification),
    'biobert': ('dmis-lab/biobert-v1.1', BertTokenizer, BertForSequenceClassification),
    'gpt2': ('gpt2-medium', GPT2Tokenizer, GPT2ForSequenceClassification),
    'biogpt': ('microsoft/biogpt', GPT2Tokenizer, GPT2ForSequenceClassification),
}

def load_model_and_tokenizer(model_name, training_size, device):
    model_path, tokenizer_class, model_class = model_map[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_path)

    if training_size == 0:
        model = model_class.from_pretrained(model_path, num_labels=6)
    else:
        model = model_class(num_labels=6)
        weights_path = f'model_{model_name}_{training_size}.bin'
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"No weights found for {model_name} with training size {training_size}.bin")

    model.to(device)
    return model, tokenizer

def get_entity_embeddings_mean(model, tokenizer, sentences_tokens, tags, layer_num=-1):
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
    
def get_data_as_tensors(df, tokenizer):
    token_ids_list = []
    tags_list = []

    for _, row in df.iterrows():
        tokens = row['tokens']
        tags = row['tags']

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        tags_tensor = torch.tensor(tags, dtype=torch.long)

        token_ids_list.append(token_ids_tensor)
        tags_list.append(tags_tensor)

    token_ids_padded = pad_sequence(token_ids_list, batch_first=True)
    tags_padded = pad_sequence(tags_list, batch_first=True)

    return token_ids_padded, tags_padded
    
def main(model_name, training_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

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

    for df in [train_df, validation_df, test_df]:
        df['tags'] = df['tags'].apply(lambda tags_list: update_tags(tags_list, label_mapping))

    model, tokenizer = load_model_and_tokenizer(model_name, training_size, device)
    train_tokens, train_tags = get_data_as_tensors(train_df)
    test_tokens, test_tags = get_data_as_tensors(test_df)

    train_entity_embeddings, train_tags, _ = get_entity_embeddings_mean(model, tokenizer, train_tokens, train_tags, layer_num=-1)
    test_entity_embeddings, test_tags, _ = get_entity_embeddings_mean(model, tokenizer, test_tokens, test_tags, layer_num=-1)

    train_entity_embeddings = torch.tensor(train_entity_embeddings, dtype=torch.float).to(device)
    train_tags = torch.tensor(train_tags, dtype=torch.long).to(device)
    test_entity_embeddings = torch.tensor(test_entity_embeddings, dtype=torch.float).to(device)
    test_tags = torch.tensor(test_tags, dtype=torch.long).to(device)

    results = {}
    for layer_num in range(0, 13):
        print(f"Probing layer {layer_num}...")
        f1_macro, f1_micro = probing_task(
            train_entity_embeddings,
            train_tags,
            test_entity_embeddings,
            test_tags
        )
        results[layer_num] = {
            "F1 Macro": f1_macro,
            "F1 Micro": f1_micro
        }

        print(f"Layer {layer_num}: F1 Macro = {f1_macro}, F1 Micro = {f1_micro}")
    print("Completed probing across all layers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and potentially fine-tune pre-trained models based on specified training size.")
    parser.add_argument("--model_name", type=str, choices=['bert', 'biobert', 'gpt2', 'biogpt'], required=True, help="The model name to use.")
    parser.add_argument("--training_size", type=int, choices=[0, 10, 30, 50, 100], required=True, help="Percentage of training data to use (0 for pre-trained model only).")

    args = parser.parse_args()

    main(args.model_name, args.training_size)
