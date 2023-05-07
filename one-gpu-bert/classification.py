# Text Classification Using BERT

# Adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc

from datasets import load_dataset

import matplotlib.pyplot as plt
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
    return load_dataset("DeveloperOats/DBPedia_Classes")

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

    return labels, id2label, label2id

def preprocess_data(samples, labels):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = samples["text"]
    encoded_text = tokenizer(text, padding="max_length", truncation=True, max_length=128)

# class Trainer:
#     def __init__(self):
#         self.num_classes = 70

def main():
    dataset = load_data()
    labels, id2label, label2id = preprocess_labels(dataset)
    # encoded_dataset = dataset.map(
    #     lambda x : preprocess_data(x, labels), 
    #     batched=True, 
    #     remove_columns=dataset["train"].column_names
    # )
    # trainer = Trainer()

main()
    