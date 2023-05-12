# Distributed Training
By Uma Bahl and Ryan Friberg

## Description
With model sizes increasing, training times have increased as well. We explore different methods of Distributed Training for two transformer-based models: BERT and ViT. This repository has the software neeed to probe distributed training with these two large models and quantify the benefits of such systems. We leveraged the code in this repository to investigate deep distributed learning as a whole but also the effects of varying the hyper parameters and georgraphic spread of the compute resources.

## Code Overview

### bert.py and vit.py 
These files respectively are the functions to call with the commands below to train the two models. Both extract and build the models and datasets from HuggingFace's Hub and construct the training task to be run from the command line.

### logcallback.py 
This file a custom HuggingFacce callback class used by both bert.py and vit.py for the purpose of collecting as much data about the training process as possible for each epoch during training. At the end of training, this callback class prints the data collected during training in the respective model's directory. Information documented includes: f1, precision, accuracy, recall, samples/s, loss, and training epoch times. The data is formatted in a way that allows for easy plotting.

### plot_values.ipynb 
This file contains a simple script used for visualizing the data generated by the LogCallBack class. Examples of the results are still present in the version committed to this repository.

### bert/ and vit/ 
These are both directories with sample output data from the LogCallBack class described above that were generated over the course of our training runs.

## Requirements
`torch`, `torchvision`, `datasets`, `transformers`, `nvidia_ml_py3`

## Repository

## Commands To Run Experiments

To run the experiments with ViT, simply replace `bert.py` with `vit.py`.

### Experiment 1: Varying Batch Sizes

#### 1 GPU, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=1 --setup 1_1`

`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=4 --setup 1_1`

`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=16 --setup 1_1`

#### 4 GPUs, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=1 --setup 1_4`

`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=4 --setup 1_4`

`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=16 --setup 1_4`

#### 2 GPUs, two nodes
Choose one GPU as the main. Set <ip> to this GPU's IP. Set this node with rank 0, and the other with rank 1.

`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=1 --setup 2_2`

`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=4 --setup 2_2`

`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --setup 2_2`
  
### Experiment 2: Increasing Learning Rate
  
#### 1 GPU, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0002 --setup 1_1`
  
`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.00002 --setup 1_1`
  
#### 4 GPUs, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0002 --setup 1_4`
  
`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0002 --setup 1_4`
  
#### 2 GPUs, two nodes
`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0002 --setup 2_2`
  
`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --epochs=5 --lr=0.00002 --setup 2_2`
  
### Experiment 3: Propagation Times
  
`torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --setup 2_1`
  
 ## Results
 
![bert_epoch](https://github.com/ubahl/Distributed-Training/assets/126620398/559bf9ed-bcb6-4a31-b348-b2861a0c1610)
  
BERT performance with varied batch size over various hardware configurations

![vit_epoch](https://github.com/ubahl/Distributed-Training/assets/126620398/5ca856b9-65d5-4e3e-a041-d03676731060)
  
ViT performance with varied batch size over various hardware configurations

![prop](https://github.com/ubahl/Distributed-Training/assets/126620398/1c5e80e0-f81b-4812-92f4-d6ff9d2e1b49)
  
Results of varying the propagation delay on distributed training
  
  
![vit1](https://github.com/ubahl/Distributed-Training/assets/126620398/a212096a-30c1-42f6-9ba9-a6e450df11ce)
![vit3](https://github.com/ubahl/Distributed-Training/assets/126620398/a2c39cf2-cccc-4df0-a673-460cb6c7a216)
![vit2](https://github.com/ubahl/Distributed-Training/assets/126620398/a536144f-cf28-44cd-81a0-acd617619c70)
  
ViT performance with varied learning rate over various hardware configurations
  
![bert1](https://github.com/ubahl/Distributed-Training/assets/126620398/6602eab1-9af1-4579-8be7-79672edc069b)
![bert2](https://github.com/ubahl/Distributed-Training/assets/126620398/812bed67-429f-4cea-8e61-e438a320df0a)
![bert3](https://github.com/ubahl/Distributed-Training/assets/126620398/978e0db0-0112-4b38-a813-137ae9744a17)
  
BERT performance with varied learning rate over various hardware configurations


