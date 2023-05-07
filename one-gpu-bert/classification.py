# Text Classification Using BERT

# Adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc

from datasets import load_dataset
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# class BERTModel(torch.nn):
#     def __init__(self):
#         super(BERTClass, self).__init__()
#         self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
#         self.l2 = torch.nn.Dropout(0.3)
#         self.l3 = torch.nn.Linear(768, 6)

def load_data():
    print("===> Loading Dataset...")
    dataset = load_dataset("DeveloperOats/DBPedia_Classes")
    print()
    return dataset

def preprocess_labels(dataset):
    print("===> Preprocessing Labels...")

    # Obtain the labels.
    labels = {}
    length = 0
    for s in dataset["train"]:
        labels[s['l2']] = 1 + labels.get(s['l2'], 0)
        length += len(s["text"])

    # Explore the statistics of the dataset.
    length /= len(dataset["train"])
    print(f"Average Length: {length:.4f}")

    plt.bar(labels.keys(), labels.values())
    plt.xticks(fontsize=6, rotation=90)
    plt.savefig("distribution.png")

    # Map labels to indices
    labels = labels.keys()
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    print()
    return labels, id2label, label2id

def preprocess_data(dataset, label2id):
    print("===> Preprocessing Dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return dataset.map(
        lambda x : preprocess_data_helper(x, tokenizer, label2id), 
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    print()

def preprocess_data_helper(samples, tokenizer, label2id):
    text = samples["text"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=500)

    label_ids = [label2id[s] for s in samples['l2']]
    label_ohs = [[1.0 if i == id else 0.0 for i in range(len(label2id))] for id in label_ids]
    
    encoding["labels"] = label_ohs
    return encoding

def main():
    dataset = load_data()
    labels, id2label, label2id = preprocess_labels(dataset)
    encoded_dataset = preprocess_data(dataset, label2id)

main()
    