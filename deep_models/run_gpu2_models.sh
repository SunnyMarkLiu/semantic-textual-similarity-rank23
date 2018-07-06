#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
python train.py --model mine.MultiChannelMatch --fold 5 --batch_size 128
