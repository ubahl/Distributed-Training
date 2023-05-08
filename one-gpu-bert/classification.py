# Text Classification Using BERT

# Adapted from https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=AFWlSsbZaRLc

from datasets import load_dataset, load_from_disk, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
from torch import cuda

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

def compute_metrics(pred):
    acc = load_metric("accuracy")
    prec = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    
    res = {"accuracy": acc.compute(predictions=predictions, references=labels)["accuracy"],
           "precision": prec.compute(predictions=predictions, references=labels, average="weighted")["precision"],
           "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
           "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]}
    return res

def main():
    # Check device.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Load dataset and labels.
    id2label, label2id = get_labels("id2label.txt", "label2id.txt")
    encoded_dataset = load_from_disk("encoded_dbpedia")

    # Create model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id)
    model.to(device)

    # Training parameters.
    batch_size = 16
    metric_name = "f1"

    args = TrainingArguments(
        output_dir="bert_one_gpu",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train and evaluate the model.
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

main()
    