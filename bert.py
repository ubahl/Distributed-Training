# HPML Final Project - Distributed Training
# Uma Bahl & Ryan Friberg

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import TrainingArguments, Trainer
from torch import cuda
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
ohs = {i : [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

def load_data():
    print("===> Loading Dataset...")
    dataset = load_dataset("ag_news")

    train_idxs = np.random.randint(0, len(dataset["train"]), size=8000)
    dataset["train"] = dataset["train"].select(train_idxs)

    test_idxs = np.random.randint(0, len(dataset["test"]), size=2000)
    dataset["test"] = dataset["test"].select(test_idxs)

    return dataset

def tokenize_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=500)
    encoding["labels"] = [ohs[l] for l in examples["label"]]
    return encoding

def create_model():
    print("===> Creating Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=4
    )
    model.to(device)
    return model

def compute_metrics(p):
    preds  = p.predictions
    preds = nn.Softmax(dim=1)(torch.tensor(preds))
    preds = torch.max(preds, 1)[1]

    labels = p.label_ids
    labels = np.argmax(labels, axis=1)

    return {
        "accuracy"  : accuracy_score(labels, preds),
        "precision" : precision_score(labels, preds, average="weighted"),
        "recall"    : recall_score(labels, preds, average="weighted"),
        "f1"        : f1_score(labels, preds, average="weighted")
    }

def main():
    dataset = load_data()
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    model = create_model()

    batch_size = 16
    epochs = 1

    args = TrainingArguments(
        output_dir="bert_one_gpu",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("===> Beginning Training...")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print("===> Beginning Evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
