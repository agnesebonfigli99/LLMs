import pandas as pd
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification

# If running on a machine with a CUDA-capable GPU, otherwise use 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = []
with open('mli_train.jsonl', 'r') as file:  # Adjust the path as necessary
    for line in file:
        data.append(json.loads(line))
data = pd.DataFrame(data)
sentences1 = data['sentence1'].values
sentences2 = data['sentence2'].values
labels = data['gold_label'].values

# Encode labels as numbers
label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
labels = [label_map[label] for label in labels]

def get_embeddings(model, tokenizer, sentences1, sentences2, layer_num=-1):
    embeddings1 = []
    embeddings2 = []
    model.eval()
    model.to(device)
    for sent1, sent2 in tqdm(zip(sentences1, sentences2), total=len(sentences1)):
        inputs1 = tokenizer(sent1, return_tensors='pt', padding=True, truncation=True).to(device)
        inputs2 = tokenizer(sent2, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs1 = model(**inputs1, output_hidden_states=True)['hidden_states'][layer_num]
            outputs2 = model(**inputs2, output_hidden_states=True)['hidden_states'][layer_num]
        embeddings1.append(outputs1.mean(dim=1).cpu().numpy())
        embeddings2.append(outputs2.mean(dim=1).cpu().numpy())
    return embeddings1, embeddings2

def probing_task(embeddings1, embeddings2, train_labels, test_embeddings1, test_embeddings2, test_labels):
    train_features = [np.concatenate((emb1.squeeze(), emb2.squeeze())) for emb1, emb2 in zip(embeddings1, embeddings2)]
    test_features = [np.concatenate((emb1.squeeze(), emb2.squeeze())) for emb1, emb2 in zip(test_embeddings1, test_embeddings2)]
    clf = SVC()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:\n", cm)
    cr = classification_report(test_labels, predictions)
    print("Classification Report:\n", cr)
    return cm, cr

train_sentences1, test_sentences1, train_sentences2, test_sentences2, train_labels, test_labels = train_test_split(sentences1, sentences2, labels, test_size=0.2, random_state=42)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
gpt_model = GPT2ForSequenceClassification.from_pretrained('gpt2-medium', num_labels=3).to(device)

folder_path = './embeddings'  # Specify your folder path here
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for layer_num in range(13):  # For BERT
    print(f"Evaluating BERT layer {layer_num}")
    bert_embeddings1, bert_embeddings2 = get_embeddings(bert_model, bert_tokenizer, train_sentences1, train_sentences2, layer_num)
    test_bert_embeddings1, test_bert_embeddings2 = get_embeddings(bert_model, bert_tokenizer, test_sentences1, test_sentences2, layer_num)

    # Save embeddings to numpy files
    np.save(os.path.join(folder_path, f'bert_embeddings1_layer_{layer_num}.npy'), np.array(bert_embeddings1))
    np.save(os.path.join(folder_path, f'bert_embeddings2_layer_{layer_num}.npy'), np.array(bert_embeddings2))
