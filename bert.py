# HPML Final Project - Distributed Training
# Uma Bahl & Ryan Friberg

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import TrainingArguments, Trainer
from torch import cuda
from torch.utils.data import DataLoader

import argparse
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Distributed Training')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--train_size', default=8000, type=int, help='size of training set')
parser.add_argument('--test_size', default=2000, type=int, help='size of test set')
parser.add_argument('--epochs', default=2, type=int, help='numper of training epochs')
parser.add_argument('--per_gpu_batch', "--b", default=16, type=int, help='batch size on each GPU')
parser.add_argument('--output_dir', "--o", default="./bert", type=str, help='output directory for model')
parser.add_argument('--grad_acc', default=4, type=int, help='gradient accumulation steps')
parser.add_argument('--run', default=1, type=int, help='run number')
parser.add_argument('--log_every', default=50, type=int, help='how often to log during training')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
ohs = {i : [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device:", device)

def load_data(args):
    print("===> Loading Dataset...")
    dataset = load_dataset("ag_news")

    train_idxs = np.random.randint(0, len(dataset["train"]), size=args.train_size)
    dataset["train"] = dataset["train"].select(train_idxs)

    test_idxs = np.random.randint(0, len(dataset["test"]), size=args.test_size)
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
    preds = p.predictions
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
    args = parser.parse_args()

    dataset = load_data(args)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    model = create_model()

    args = TrainingArguments(
        output_dir=f"{args.output_dir}_{args.run}",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_gpu_batch,
        per_device_eval_batch_size=args.per_gpu_batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=args.log_every,
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
