import tensorflow as tf
from collections import OrderedDict

class VisualBackPropModel():
    def _aver_feature_layer(self,
                            _layer):
        return tf.reduce_mean(_layer, axis = -1, keep_dims = True)

    def __init__(self,
                 _feature_layers,
                 _kernel_sizes,
                 _strides):
        self.feature_layers = _feature_layers

        # ===== Average featuree layers =====
        self.aver_feature_layers = [self._aver_feature_layer(_feature_layer)
                                    for _feature_layer in _feature_layers]

        # ===== Visual Backprop =====
        self.visual_layers = [self.aver_feature_layers[-1]]
        self.layers        = []
        for _feature_layer, _kernel_size, _stride in zip(reversed(self.aver_feature_layers[:-1]), reversed(_kernel_sizes), reversed(_strides)):
            _transposed_conv = tf.layers.conv2d_transpose(inputs             = self.visual_layers[0],
                                                          filters            = 1,
                                                          kernel_size        = _kernel_size,
                                                          strides            = _stride,
                                                          padding            = 'same',
                                                          kernel_initializer = tf.ones_initializer())
            self.layers.insert(0, _transposed_conv)
            self.visual_layers.insert(0, _feature_layer * _transposed_conv)
        # Last layer
        _transposed_conv = tf.layers.conv2d_transpose(inputs             = self.visual_layers[0],
                                                      filters            = 1,
                                                      kernel_size        = _kernel_sizes[0],
                                                      strides            = _strides[0],
                                                      padding            = 'same',
                                                      kernel_initializer = tf.ones_initializer())
        self.layers.insert(0, _transposed_conv)
        self.visual_layers.insert(0, _transposed_conv)

        def _visual_func(_session, _state,
                         _batch_x):
            return _session.run(self.visual_layers,
                                feed_dict = {
                                    'state:0':  _state,
                                    'input:0':  _batch_x
                                })
        self.visual_func = _visual_func




