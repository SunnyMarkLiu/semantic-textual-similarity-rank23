#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-23 下午3:50
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import warnings
from keras.callbacks import Callback
from keras import backend as K


class DynamicLayerTrainable(Callback):
    """ set model layer trainable or not trainable dynamically """
    def __init__(
            self,
            monitor='val_loss',
            set_layer_index=None,
            show_all_layers=False,
            verbose=0,
            mode='auto'
    ):
        super(DynamicLayerTrainable, self).__init__()
        self.monitor = monitor
        self.set_layer_index = set_layer_index
        self.show_all_layers = show_all_layers
        self.set_layer_trainable = False
        self.verbose = verbose

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        if self.show_all_layers:
            print('='*20)
            print(self.model.layers)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):

                self.best = current
                self.best_epoch = epoch
                # clear patience_index
                self.patience_index = 0

                # clear flag
                self.continus_no_improvet_epoch = 0

            else:
                # update layer trainable true
                if epoch > 0:
                    # didn't improve
                    # 第一次设置 variable 的 trainable 为 True
                    if not self.set_layer_trainable:
                        self.set_layer_trainable = True
                        set_layer = self.model.layers[self.set_layer_index]

                        if self.verbose:
                            print('\nset embedding layer trainable from {} to {}'.format(set_layer.trainable,
                                                                                       not set_layer.trainable))
                        set_layer.trainable = not set_layer.trainable


class ModelSave_EarlyStop_LRDecay(Callback):
    """Save the model after every epoch.

    `model_path` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `model_path` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        model_path: string, path to save the model file.
        lr_decay: learning rate decay
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(model_path)`), else the full model
            is saved (`model.save(model_path)`).
        period: Interval (number of epochs) between checkpoints.
        patience: spectfic the count that `monitor` didn't improve, then update learning reate
    """

    def __init__(self, model_path, lr_decay=0.9, monitor='val_loss', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 patience_continus_no_improvet_epoch=2,
                 mode='auto', period=1, patience=3):
        super(ModelSave_EarlyStop_LRDecay, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.model_path = model_path
        self.lr_decay = lr_decay
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.patience = patience
        self.patience_index = 0

        # continues updating lr twice no improvement，stop training
        self.patience_continus_no_improvet_epoch = patience_continus_no_improvet_epoch
        self.continus_no_improvet_epoch = 0

        self.best_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            model_path = self.model_path.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' %
                                  (epoch, self.monitor, self.best, current))
                        self.best = current
                        self.best_epoch = epoch
                        # clear patience_index
                        self.patience_index = 0
                        if self.save_weights_only:
                            self.model.save_weights(model_path, overwrite=True)
                        else:
                            self.model.save(model_path, overwrite=True)

                        # clear flag
                        self.continus_no_improvet_epoch = 0

                    else:
                        # update learning rate
                        if epoch > 0:
                            # didn't improve
                            self.patience_index += 1
                            if self.patience_index >= self.patience:
                                old_lr = K.get_value(self.model.optimizer.lr)
                                new_lr = self.lr_decay * old_lr
                                K.set_value(self.model.optimizer.lr, new_lr)
                                print('\nEpoch %05d: %s did not improve' %
                                      (epoch, self.monitor))
                                print('Epoch %05d: change learning rate from %f to %f'
                                      % (epoch, old_lr, K.get_value(self.model.optimizer.lr)))
                                # clear patience_index
                                self.patience_index = 0

                            self.continus_no_improvet_epoch += 1

                    if self.continus_no_improvet_epoch > self.patience_continus_no_improvet_epoch:
                        print('\nStop training... \nbest %s : %0.5f, epoch: %d, model path: %s' %
                              (self.monitor, self.best, self.best_epoch, self.model_path))
                        self.model.stop_training = True

            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch, model_path))
                if self.save_weights_only:
                    self.model.save_weights(model_path, overwrite=True)
                else:
                    self.model.save(model_path, overwrite=True)
