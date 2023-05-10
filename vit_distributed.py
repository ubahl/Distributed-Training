
import argparse
from datasets import load_dataset, load_metric
import numpy as np
import nvidia_smi
import time
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, DefaultDataCollator, TrainerCallback

'''
How to run:
torchrun  \
    --nproc_per_node= however many GPUs are on each node \
    --nnodes= number of nodes \
    --node_rank= $THIS_MACHINE_INDEX \
    --master_addr=... \
    --master_port=1234 \
    vit_distributed.py \
    (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)
'''

parser = argparse.ArgumentParser(description='Distributed Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--test_split', default=0.2, type=float, help='percentage of dataset to be used for training')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--per_gpu_batch', "--b", default=16, type=int, help='batch size on each GPU')
parser.add_argument('--output_dir', "--o", default="./vit", type=str, help='batch size on each GPU')
parser.add_argument('--grad_acc', default=4, type=int, help='gradient accumulation steps')
parser.add_argument('--setup', default="1_1", type=str, help='number of nodes then number of gpus')
# parser.add_argument('--warm_up', default=0.1, type=float, help='warm up ratio')

model_name = 'google/vit-base-patch16-224-in21k'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
img_transforms = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
                                     transforms.Resize((224,224))])


class LogCallBack(TrainerCallback):
    def __init__(self, trainer, run_name):
        self._trainer = trainer
        self.f1    = []
        self.acc   = []
        self.prec  = []
        self.rec   = []
        self.times = []
        self.loss  = []
        self.sps   = []
        self.run_name = run_name

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.run_name, 'w+') as out_file:
            out_file.write("f1:[", ",".join(self.f1), "]")
            out_file.write("acc:[", ",".join(self.acc), "]")
            out_file.write("prec:[", ",".join(self.prec), "]")
            out_file.write("rec:[", ",".join(self.rec), "]")
            out_file.write("times:[", ",".join(self.times), "]")
            out_file.write("loss:[", ",".join(self.loss), "]")
            out_file.write("samples/s:[", ",".join(self.sps), "]")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.f1.append(metrics['eval_f1'])
        self.acc.append(metrics['eval_acc'])
        self.prec.append(metrics['eval_prec'])
        self.rec.append(metrics['eval_recall'])
        self.loss.append(metrics['eval_loss'])
        self.sps.append(metrics['eval_samples_per_second'])

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)

def print_gpu_utilization():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, 
            nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()

def transform_data(examples):
    examples["pixel_values"] = [img_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

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
    args = parser.parse_args()

    run_name = args.output_dir + "/" + args.setup + "_" + str(args.lr) + "_" + str(args.per_gpu_batch)
    print("Saving metics to", run_name)

    print("\nBuilding dataset...")
    food_dataset = load_dataset("food101", split="train[:10000]")
    food_dataset = food_dataset.train_test_split(test_size=args.test_split)
    food_dataset = food_dataset.with_transform(transform_data)

    labels = food_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print("\nBuilding model...")
    model = ViTForImageClassification.from_pretrained(model_name,
                                                    num_labels=len(labels),
                                                    id2label=id2label,
                                                    label2id=label2id)
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=run_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_gpu_batch,
        gradient_accumulation_steps=args.grad_acc,
        per_device_eval_batch_size=args.per_gpu_batch,
        num_train_epochs=args.epochs,
        # warmup_ratio=args.warm_up,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=food_dataset["train"],
        eval_dataset=food_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(LogCallBack(trainer, run_name)) 

    print("\nBeginning training...")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    print("\nBeginning eval...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__=='__main__':
    main()

