# MultiChannelMatch 调试记录

# 三个子模型拼接方式的模型复杂度选择
# simple，泛化能力似乎不错

    ============== perform fold 0, total folds 5 ==============
    Train on 203508 samples, validate on 50878 samples
    Epoch 1/100
    203508/203508 [==============================] - 156s 769us/step - loss: 0.4161 - binary_accuracy: 0.8096 - val_loss: 0.3329 - val_binary_accuracy: 0.8599
    Epoch 2/100
    203508/203508 [==============================] - 154s 757us/step - loss: 0.3600 - binary_accuracy: 0.8410 - val_loss: 0.2999 - val_binary_accuracy: 0.8734
    Epoch 3/100
    203508/203508 [==============================] - 152s 747us/step - loss: 0.3408 - binary_accuracy: 0.8510 - val_loss: 0.2811 - val_binary_accuracy: 0.8812
    Epoch 4/100
    203508/203508 [==============================] - 153s 754us/step - loss: 0.3285 - binary_accuracy: 0.8567 - val_loss: 0.2720 - val_binary_accuracy: 0.8869
    Epoch 5/100
    203508/203508 [==============================] - 154s 755us/step - loss: 0.3180 - binary_accuracy: 0.8620 - val_loss: 0.2704 - val_binary_accuracy: 0.8872
    Epoch 6/100
    203508/203508 [==============================] - 154s 755us/step - loss: 0.3128 - binary_accuracy: 0.8656 - val_loss: 0.2599 - val_binary_accuracy: 0.8908
    Epoch 7/100
    203508/203508 [==============================] - 153s 752us/step - loss: 0.3054 - binary_accuracy: 0.8685 - val_loss: 0.2526 - val_binary_accuracy: 0.8931
    Epoch 8/100
    203508/203508 [==============================] - 152s 747us/step - loss: 0.3007 - binary_accuracy: 0.8707 - val_loss: 0.2500 - val_binary_accuracy: 0.8929
    Epoch 9/100
    203508/203508 [==============================] - 152s 748us/step - loss: 0.2959 - binary_accuracy: 0.8732 - val_loss: 0.2504 - val_binary_accuracy: 0.8951
    Epoch 10/100
    203508/203508 [==============================] - 150s 736us/step - loss: 0.2926 - binary_accuracy: 0.8747 - val_loss: 0.2479 - val_binary_accuracy: 0.8954
    Epoch 11/100
    203508/203508 [==============================] - 149s 734us/step - loss: 0.2896 - binary_accuracy: 0.8767 - val_loss: 0.2457 - val_binary_accuracy: 0.8968
    Epoch 12/100
    203508/203508 [==============================] - 149s 732us/step - loss: 0.2875 - binary_accuracy: 0.8767 - val_loss: 0.2414 - val_binary_accuracy: 0.8994
    Epoch 13/100
    203508/203508 [==============================] - 149s 731us/step - loss: 0.2858 - binary_accuracy: 0.8775 - val_loss: 0.2393 - val_binary_accuracy: 0.9005
    Epoch 14/100
    203508/203508 [==============================] - 149s 731us/step - loss: 0.2817 - binary_accuracy: 0.8790 - val_loss: 0.2404 - val_binary_accuracy: 0.8999
    Epoch 15/100
    203508/203508 [==============================] - 149s 732us/step - loss: 0.2819 - binary_accuracy: 0.8797 - val_loss: 0.2342 - val_binary_accuracy: 0.9014
    Epoch 16/100
    203508/203508 [==============================] - 149s 731us/step - loss: 0.2771 - binary_accuracy: 0.8829 - val_loss: 0.2365 - val_binary_accuracy: 0.9008
    Epoch 17/100
    203508/203508 [==============================] - 150s 735us/step - loss: 0.2743 - binary_accuracy: 0.8841 - val_loss: 0.2326 - val_binary_accuracy: 0.9026

# complicated，性能似乎更好

    ============== perform fold 0, total folds 5 ==============
    Train on 203508 samples, validate on 50878 samples
    Epoch 1/100
    203508/203508 [==============================] - 1238s 6ms/step - loss: 0.3806 - binary_accuracy: 0.8303 - val_loss: 0.3449 - val_binary_accuracy: 0.8553
    Epoch 2/100
     52928/203508 [======>.......................] - ETA: 14:29 - loss: 0.3156 - binary_accuracy: 0.8656^@
