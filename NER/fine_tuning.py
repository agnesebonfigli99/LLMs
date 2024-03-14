
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


def load_data(training_size):
    dataset = load_dataset("tner/bionlp2004")
    train_df = pd.DataFrame({'tokens': dataset['train']['tokens'], 'tags': dataset['train']['tags']})
    validation_df = pd.DataFrame({'tokens': dataset['validation']['tokens'], 'tags': dataset['validation']['tags']})
    test_df = pd.DataFrame({'tokens': dataset['test']['tokens'], 'tags': dataset['test']['tags']})

    if training_size < 100:
        train_df = train_df.sample(frac=training_size / 100)

    return train_df, validation_df, test_df 

def load_model_and_tokenizer(model_name, training_size, device):
    model_map = {
        'bert': ('bert-base-uncased', BertTokenizer, BertForTokenClassification),
        'biobert': ('dmis-lab/biobert-v1.1', BertTokenizer, BertForTokenClassification),
        'gpt2': ('gpt2-medium', GPT2Tokenizer, GPT2ForTokenClassification),
        'biogpt': ('microsoft/biogpt', GPT2Tokenizer, GPT2ForTokenClassification),  # Assumendo che GPT2Tokenizer & Classifier siano corretti per biogpt
    }

    model_path, tokenizer_class, model_class = model_map[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_path)

    if training_size == 0:
        model = model_class.from_pretrained(model_path, num_labels=6, output_attentions=True)
    else:
        path_to_finetuned_model_weights = f'/path/to/save/model_{model_name}_{training_size}.bin'
        model = model_class(num_labels=6, output_attentions=True)
        state_dict = torch.load(path_to_finetuned_model_weights, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, tokenizer

def main(model_name, training_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer, model = load_model_and_tokenizer(model_name)
    model.to(device)

    if training_size > 0:
        train_df, validation_df, test_df = load_data(training_size / 100.0)

        train_df['tags'] = train_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))
        validation_df['tags'] = validation_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))
        test_df['tags'] = test_df['tags'].apply(lambda tags_list: update_tags(tags_list, updated_label_mapping))

        train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
        valid_dataset = CustomDataset(validation_df, tokenizer, MAX_LEN)
        test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            train(epoch, train_loader, model, optimizer)

        print("Training completed. Validating the model...")
        valid(model, valid_loader)
    else:
        print("Training size set to 0. Skipping fine-tuning and using the pre-trained model directly.")

    # Salvataggio del modello
    save_path = f'model_{model_name}_{training_size}.bin'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["bert", "gpt2", "biobert", "biobert"], required=True, help="Model name")
    parser.add_argument("--training_size", type=int, choices=[0, 10, 30, 50, 100], default=100, help="Percentage of training data to use (0 for pre-trained model only)")

    args = parser.parse_args()

    main(args.model_name, args.training_size)


