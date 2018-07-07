#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/7/6 下午9:56
"""
import math


def divide_decay(epoch, current_lr):
    """
    LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
    :param epoch: 当前 epoch
    :return:    当前 lr
    """
    drop = 0.5
    new_lr = current_lr * drop
    return new_lr
