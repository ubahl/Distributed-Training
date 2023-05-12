# Distributed Training
By Uma Bahl and Ryan Friberg

## Description
With model sizes increasing, training times have increased as well. We explore different methods of Distributed Training for two transformer-based models: BERT and ViT.

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
