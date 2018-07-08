#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/7/8 上午11:28
"""
from keras.layers import *
from keras.activations import sigmoid


def gated_liner_units(input1, input2, filters, kernel_size, padding, activation):
    """
    Ref: Language Modeling with Gated Convolutional Networks

    ∇[X⊗σ(X)]=∇X⊗σ(X)+X⊗σ′(X)∇X
    """
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size,
                        padding=padding, activation=activation)
    conv1 = conv_layer(input1)
    conv2 = conv_layer(input2)

    gate = Lambda(lambda x: sigmoid(x))(conv2)
    gate_out_conv1 = Multiply()([conv1, gate])
    # residual connect
    conv1 = Add()([conv1, gate_out_conv1])

    gate = Lambda(lambda x: sigmoid(x))(conv1)
    gate_out_conv2 = Multiply()([conv2, gate])
    conv2 = Add()([conv2, gate_out_conv2])

    return conv1, conv2
