#!/Users/sunnymarkliu/softwares/miniconda3/bin/python
# _*_ coding: utf-8 _*_

"""
@author: SunnyMarkLiu
@time  : 2018/7/6 下午8:49
"""
import warnings

import numpy as np
from keras.callbacks import Callback
from keras import backend as K


class ModelCheckPointByBatch_EarlyStop_LRScheduler(Callback):
    """
    完成三个功能：

    1、检测每个 epoch 的数据训练之后，valid 的 loss，依据此保存最优模型参数，比依据 epoch 要细粒度
    2、检测每个 epoch 的数据训练之后，如果多次还没有提示则 early stop
    3、如果多次 epoch 之后还没有提升，则加载监测 epoch 的最优模型参数，再 lr scheduler
    """

    def __init__(self, best_model_path, monitor='val_loss', train_monitor='loss', period=1,
                 save_best_only=True, save_weights_only=False,
                 valid_batch_size=None, validation_data=None, metric_fun=None,
                 valid_batch_interval=None,
                 stop_patience_epoch=3, stop_min_delta=0.,
                 lr_schedule_patience_epoch=2,
                 schedule_fun=None,
                 mode='auto', verbose=0):
        super(ModelCheckPointByBatch_EarlyStop_LRScheduler, self).__init__()

        ############### model checkpoint params ###############
        self.monitor = monitor
        self.train_monitor = train_monitor
        self.verbose = verbose
        self.best_model_path = best_model_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.epochs_since_last_save = 0
        self.period = period

        self.valid_x = validation_data[0]
        self.valid_y = validation_data[1]
        self.metric_fun = metric_fun
        self.valid_batch_size = valid_batch_size
        self.valid_batch_interval = valid_batch_interval
        self.previous_valid_batch = 0

        ############### early cstopping params ###############
        self.stop_patience_epoch = stop_patience_epoch
        self.min_delta = stop_min_delta
        self.wait = 0
        self.stopped_epoch = 0

        ############### lr schedule params ###############
        self.lr_schedule_patience_epoch = lr_schedule_patience_epoch  # 在 early stop 之前再降低 lr 挣扎下
        self.schedule = schedule_fun

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
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

    ############### model checkpoint ###############
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            best_model_path = self.best_model_path.format(batch + 1, **logs)
            if self.save_best_only:

                # 手动验证验证集的结果
                if batch - self.previous_valid_batch < self.valid_batch_interval:
                    return

                valid_pred = self.model.predict(self.valid_x, batch_size=self.valid_batch_size)[:, 0]
                current = self.metric_fun(self.valid_y, valid_pred)
                self.previous_valid_batch = batch

                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('batch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (batch + 1, self.monitor, self.best,
                                     current, best_model_path))
                        self.best = current
                        self.train_best = logs.get(self.train_monitor)
                        # clear early stop
                        self.wait = 0

                        if self.save_weights_only:
                            self.model.save_weights(best_model_path, overwrite=True)
                        else:
                            self.model.save(best_model_path, overwrite=True)
                    else:
                        pass
                        # if self.verbose > 0:
                        #     print('batch %05d: %s did not improve' % (batch + 1, self.monitor))

            else:
                # if self.verbose > 0:
                #     print('batch %05d: saving model to %s' % (batch + 1, best_model_path))
                if self.save_weights_only:
                    self.model.save_weights(best_model_path, overwrite=True)
                else:
                    self.model.save(best_model_path, overwrite=True)

    ############### early stopping ###############
    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):

        self.previous_valid_batch = 0

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait == self.lr_schedule_patience_epoch:
                ############### LR Scheduler ###############
                if not hasattr(self.model.optimizer, 'lr'):
                    raise ValueError('Optimizer must have a "lr" attribute.')

                # 1、加载 epoch 训练之后的最优模型参数
                self.model.load_weights(filepath=self.best_model_path)

                # 2、重新设置 lr
                current_lr = K.get_value(self.model.optimizer.lr)
                lr = self.schedule(epoch, current_lr)  # 传入当前 epoch 和 lr
                if not isinstance(lr, (float, np.float32, np.float64)):
                    raise ValueError('The output of the "schedule" function '
                                     'should be float.')

                if self.verbose > 0:
                    print('Epoch {}: lr decay from {} to {}'.format(self.stopped_epoch + 1, current_lr, lr))

                K.set_value(self.model.optimizer.lr, lr)

            # lr 调整之后，还是不行，则 stop
            if self.wait >= self.stop_patience_epoch:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
