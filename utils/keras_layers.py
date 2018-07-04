#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/6/29 下午3:26
"""
from keras.layers import *
from keras.activations import softmax


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def substract(input_1, input_2):
    """Substract element-wise"""
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def diff_features(input_1, input_2):
    sub = substract(input_1, input_2)
    mult = Multiply()([input_1, input_2])

    features = Concatenate()([sub, mult])
    features = BatchNormalization()(features)
    return features


def apply_multiple(input_, layers):
    """Apply layers to input then concatenate result"""
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    """Apply a list of layers in TimeDistributed mode"""
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned
