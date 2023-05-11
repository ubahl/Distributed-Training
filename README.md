# Distributed Training

## BERT

### Experiment 1: Strong Scaling vs. Weak Scaling

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
  
### Experiment 2: Learning Rate
  
#### 1 GPU, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0001 --setup 1_1`
  
`torchrun --standalone --nnodes=1 --nproc_per_node=1 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.00001 --setup 1_1`
  
#### 4 GPUs, one node
`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0001 --setup 1_4`
  
`torchrun --standalone --nnodes=1 --nproc_per_node=4 bert.py --per_gpu_batch=16 --epochs=5 --lr=0.00001 --setup 1_4`
  
#### 2 GPUs, two nodes
`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --epochs=5 --lr=0.0001 --setup 2_2`
  
`torchrun --nnodes=2 --nproc_per_node=2 --node_rank=<0,1> --master_addr=<ip> --master_port=<port> bert.py --per_gpu_batch=16 --epochs=5 --lr=0.00001 --setup 2_2`
