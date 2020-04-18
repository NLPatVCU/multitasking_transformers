#!/bin/bash

source ~/testing/multitasking_transformers/bin/activate

echo $PATH
export PYTHONPATH=${PYTHONPATH}:.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64/
python -V
CUDA_VISIBLE_DEVICES=6 python predict.py