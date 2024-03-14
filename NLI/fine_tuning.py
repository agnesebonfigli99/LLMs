import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import (BertTokenizer, BertForSequenceClassification,
                          GPT2Tokenizer, GPT2ForSequenceClassification,
                          AutoTokenizer, AutoModelForSequenceClassification)
import json
import pandas as pd
from tqdm import tqdm

def prepare_data(sentences1, sentences2, labels, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sentences1.tolist(), sentences2.tolist(), padding=True, truncation=True, return_tensors="pt")
    labels_tensor = torch.tensor(labels)
    return DataLoader(TensorDataset(inputs.input_ids, inputs.attention_mask, labels_tensor), batch_size=1)

def main(training_size, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    data = []
    with open('/.../mli_train.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))

    data = pd.DataFrame(data)

    # Map labels to numbers
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    data['gold_label'] = data['gold_label'].map(label_map)

    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['gold_label'].values

    # Split the data
    train_size = training_size / len(data) if training_size > 0 else 0.9  # Adjust based on training size input
    test_size = 1 - train_size
    train_sentences1, test_sentences1, train_labels, test_labels = train_test_split(sentences1, labels, test_size=test_size, random_state=42, stratify=labels)
    train_sentences2, test_sentences2, _, _ = train_test_split(sentences2, labels, test_size=test_size, random_state=42, stratify=labels)

    model_map = {
        'bert': ('bert-base-uncased', BertTokenizer, BertForSequenceClassification),
        'biobert': ('dmis-lab/biobert-v1.1', BertTokenizer, BertForSequenceClassification),
        'gpt2': ('gpt2-medium', GPT2Tokenizer, GPT2ForSequenceClassification),
        'biogpt': ('microsoft/biogpt', GPT2Tokenizer, GPT2ForSequenceClassification),  # Assumption: BioGpt uses GPT2Tokenizer & Classifier
    }

    model_path, tokenizer_class, model_class = model_map[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, num_labels=3, output_attentions=True)
    
    if training_size > 0:
        train_loader = prepare_data(train_sentences1, train_sentences2, train_labels, tokenizer)
        test_loader = prepare_data(test_sentences1, test_sentences2, test_labels, tokenizer)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()

        # Fine-tuning
        print("Starting fine-tuning...")
        model.train()
        for epoch in range(3):
            total_loss = 0
            for batch in tqdm(train_loader):
                batch = [b.to(device) for b in batch]
                input_ids, attention_mask, labels = batch

                model.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")

    # Evaluation could go here if training_size > 0, or with pretrained model as desired

    torch.save(model.state_dict(), f'/path/to/save/model_{model_name}.bin') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a model on a dataset')
    parser.add_argument('--training_size', type=int, choices=[0, 10, 30, 50, 100], help='Size of the training set')
    parser.add_argument('--model_name', type=str, choices=['bert', 'biobert', 'gpt2', 'biogpt'], help='Model to fine-tune')
    
    args = parser.parse_args()
    main(args.training_size, args.model_name)
