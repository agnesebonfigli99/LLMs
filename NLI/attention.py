#!pip install datasets transformers scikit-learn pandas
import json
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, GPT2Tokenizer, BioGptTokenizer, BertForSequenceClassification, GPT2ForSequenceClassification, BioGptForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium').eos_token
    tokenizer = BertTokenizer.from_pretrained('mis-lab/biobert-v1.1')
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt").eos_token

    model_pretrained = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=True)
    model_pretrained = GPT2ForSequenceClassification.from_pretrained('gpt2-medium', num_labels=3, output_attentions=True)
    model_pretrained = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=3, output_attentions=True)
    model_pretrained = BioGptForSequenceClassification.from_pretrained('microsoft/biogpt', num_labels=3, output_attentions=True)
    model_pretrained = model_pretrained.to(device)
    model_pretrained.eval()

    model_finetuned = GPT2ForSequenceClassification.from_pretrained('gpt2-medium', num_labels=3, output_attentions=True)
    path_to_finetuned_model_weights = '/content/drive/MyDrive/fine_tunedNLI_GPT10_model.pth'
    state_dict = torch.load(path_to_finetuned_model_weights, map_location=device)
    model_finetuned.load_state_dict(state_dict)
    model_finetuned = model_finetuned.to(device)
    model_finetuned.eval() 

    data = []
    with open('/content/drive/MyDrive/mli_train.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))
    data = pd.DataFrame(data)
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['label'].values

    _, test_sentences1, _, test_sentences2, _, _ = train_test_split(sentences1, sentences2, labels, test_size=0.2, random_state=42)

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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_sim_layers, annot=True, fmt=".2f", cmap='viridis', xticklabels=range(1, num_attention_heads+1), yticklabels=range(1, num_layers+1))
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.title("Average Cosine Similarity between Pre-trained and Fine-tuned Models")
    plt.show()

if __name__ == "__main__":
    main()
