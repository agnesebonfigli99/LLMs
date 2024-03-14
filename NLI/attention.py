#!pip install datasets transformers scikit-learn pandas
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, GPT2Tokenizer, BertForSequenceClassification, GPT2ForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_tokenizer(model_name, training_size, device):
    model_map = {
        'bert': ('bert-base-uncased', BertTokenizer, BertForSequenceClassification),
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

def main(model_name, training_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricare i modelli e i tokenizer
    model_pretrained, _ = load_model_and_tokenizer(model_name, 0, device)
    model_finetuned, tokenizer = load_model_and_tokenizer(model_name, training_size, device)

    # Caricare e preparare i dati
    data = []
    with open('/content/drive/MyDrive/mli_train.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))
    data = pd.DataFrame(data)
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['label'].values

    _, test_sentences1, _, test_sentences2, _, _ = train_test_split(sentences1, sentences2, labels, test_size=0.2, random_state=42)

    # Calcolare la similarità coseno tra le attenzioni dei modelli
    num_layers = model_pretrained.config.num_hidden_layers
    num_attention_heads = model_pretrained.config.num_attention_heads
    cos_sim_layers = np.zeros((num_layers, num_attention_heads))

    for s1, s2 in zip(test_sentences1, test_sentences2):
        encoded_input = tokenizer(s1, s2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            outputs_pretrained = model_pretrained(**encoded_input, output_attentions=True)
            outputs_finetuned = model_finetuned(**encoded_input, output_attentions=True)

        for layer in range(num_layers):
            for head in range(num_attention_heads):
                attn_pretrained = outputs_pretrained.attentions[layer][0, head].detach().cpu().numpy()
                attn_finetuned = outputs_finetuned.attentions[layer][0, head].detach().cpu().numpy()

                attn_pretrained_flat = attn_pretrained.flatten()
                attn_finetuned_flat = attn_finetuned.flatten()
                cos_sim = cosine_similarity([attn_pretrained_flat], [attn_finetuned_flat])[0][0]
                cos_sim_layers[layer, head] += cos_sim

    cos_sim_layers /= len(test_sentences1)

    # Visualizzare la similarità coseno
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_sim_layers, annot=True, fmt=".2f", cmap='viridis', xticklabels=range(1, num_attention_heads+1), yticklabels=range(1, num_layers+1))
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.title("Average Cosine Similarity between Pre-trained and Fine-tuned Models")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model attentions between pre-trained and fine-tuned models.")
    parser.add_argument('--training_size', type=int, choices=[0, 10, 30, 50, 100], help='Size of the training set as a percentage')
    parser.add_argument('--model_name', type=str, choices=['bert', 'biobert', 'gpt2', 'biogpt'], help='Model to fine-tune')
    
    args = parser.parse_args()
    main(args.model_name, args.training_size)
