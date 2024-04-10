#!pip install datasets transformers scikit-learn pandas
import argparse
import json
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, GPT2Tokenizer, BioGptTokenizer, BertForTokenClassification, GPT2ForTokenClassification, BioGptForTokenClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def load_model_and_tokenizer(model_name, training_size, device):
    model_map = {
        'bert': ('bert-base-cased', BertTokenizer, BertForTokenClassification),
        'biobert': ('dmis-lab/biobert-v1.1', BertTokenizer, BertForTokenClassification),
        'gpt2': ('gpt2-medium', GPT2Tokenizer, GPT2ForTokenClassification),
        'biogpt': ('microsoft/biogpt', GPT2Tokenizer, GPT2ForTokenClassification),
    }

    model_path, tokenizer_class, model_class = model_map[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_path)

    if training_size == 0:
        model = model_class.from_pretrained(model_path, num_labels=3, output_attentions=True)
    else:
        path_to_finetuned_model_weights = f'/path/to/model_{model_name}_{training_size}.bin'
        model = model_class(num_labels=3, output_attentions=True)
        state_dict = torch.load(path_to_finetuned_model_weights, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, tokenizer  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_name, training_size, device)
    dataset = load_dataset("tner/bionlp2004")
    def dataset_to_df(split):
        return pd.DataFrame({'tokens': dataset[split]['tokens'], 'tags': dataset[split]['tags']})
    train_df = dataset_to_df('train')
    validation_df = dataset_to_df('validation')
    test_df = dataset_to_df('test')
    
    def update_tags(tags_list, label_mapping):
        return [label_mapping[tag] if tag in label_mapping else tag for tag in tags_list]
    
    label_mapping = {
        "O": 0,
        "B-DNA": 1, "I-DNA": 1,  
        "B-protein": 2, "I-protein": 2,  
        "B-cell_type": 3, "I-cell_type": 3,  
        "B-cell_line": 4, "I-cell_line": 4,  
        "B-RNA": 5, "I-RNA": 5  
    }
    
    for df in [train_df, validation_df, test_df]:
        df['tags'] = df['tags'].apply(lambda tags: update_tags(tags, label_mapping))
    
    train_tokens = train_df['tokens']
    train_tags = train_df['tags']
    test_tokens = test_df['tokens']
    test_tags = test_df['tags']
    cos_sim_layers = np.zeros((num_layers, num_attention_heads))
    
    for s in test_tokens:
        sentence = " ".join(tokens)
        encoded_input = tokenizer(s, return_tensors='pt', padding=True, truncation=True, max_length=512)
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
  
    cos_sim_layers /= len(test_tokens)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_sim_layers, annot=True, fmt=".2f", cmap='viridis', xticklabels=range(1, num_attention_heads+1), yticklabels=range(1, num_layers+1))
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.title("Average Cosine Similarity between Pre-trained and Fine-tuned Models")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model attentions between pre-trained and fine-tuned models.")
    parser.add_argument('--training_sizes', type=str, help='Sizes of the training sets as percentages, separated by commas (e.g., 10,30)')
    parser.add_argument('--model_name', type=str, choices=['bert', 'biobert', 'gpt2', 'biogpt'], help='Model to fine-tune')
    
    args = parser.parse_args()
    main(args.model_name, args.training_size)
