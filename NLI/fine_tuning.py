import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def prepare_data(sentences1, sentences2, labels, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sentences1.tolist(), sentences2.tolist(), padding=True, truncation=True, return_tensors="pt")
    labels_tensor = torch.tensor(labels)
    return DataLoader(TensorDataset(inputs.input_ids, inputs.attention_mask, labels_tensor), batch_size=1)

def main():
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
    train_sentences1, test_sentences1, train_labels, test_labels = train_test_split(sentences1, labels, test_size=0.9, random_state=42, stratify=labels)
    train_sentences2, test_sentences2, _, _ = train_test_split(sentences2, labels, test_size=0.9, random_state=42, stratify=labels)

    train_sentences1, _, train_labels, _ = train_test_split(train_sentences1, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
    train_sentences2, _, _, _ = train_test_split(train_sentences2, train_labels, test_size=0.2, random_state=42, stratify=train_labels)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium').eos_token
    tokenizer = BertTokenizer.from_pretrained('mis-lab/biobert-v1.1')
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt").eos_token

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=True)
    model = GPT2ForSequenceClassification.from_pretrained('gpt2-medium', num_labels=3, output_attentions=True)
    model= BertForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=3, output_attentions=True)
    model = BioGptForSequenceClassification.from_pretrained('microsoft/biogpt', num_labels=3, output_attentions=True)
    
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

    model.eval()
    preds, true = [], []
    for batch in tqdm(test_loader):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().tolist())
            true.extend(labels.cpu().tolist())

    torch.save(model.state_dict(), '/.../') 

if __name__ == "__main__":
    main()
