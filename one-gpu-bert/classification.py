# Text Classification Using BERT

# Adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def get_labels(id2label_file, label2id_file):
    with open(id2label_file) as f:
        id2label_data = f.read()
    
    with open(label2id_file) as f:
        label2id_data = f.read()
    
    return json.loads(id2label_data), json.loads(label2id_data)

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def main():
    id2label, label2id = get_labels("id2label.txt", "label2id.txt")
    encoded_dataset = load_from_disk("encoded_dbpedia")

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",
    #     problem_type="multi_label_classification",
    #     num_labels=len(labels),
    #     id2label=id2label,
    #     label2id=label2id)

    # batch_size = 32
    # metric_name = "f1"

    # args = TrainingArguments(
    #     f"bert-finetuned-sem_eval-english",
    #     evaluation_strategy = "epoch",
    #     save_strategy = "epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model=metric_name,
    # )

    # trainer = Trainer(
    #     model,
    #     args,
    #     train_dataset=encoded_dataset["train"],
    #     eval_dataset=encoded_dataset["validation"],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )

    # trainer.evaluate()

main()
    