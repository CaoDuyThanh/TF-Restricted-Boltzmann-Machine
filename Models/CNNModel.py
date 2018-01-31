import tensorflow as tf
from collections import OrderedDict
from Layers.Optimizer import *

class CNNModel():
    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.input         = tf.placeholder(tf.float32, shape = [None, 784], name = 'input')
        self.output        = tf.placeholder(tf.int32, shape = [None], name = 'output')
        self.batch_size    = tf.placeholder(tf.int32, shape = (), name = 'batch_size')
        self.state         = tf.placeholder(tf.bool, shape = (), name = 'state')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'learning_rate')

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Reshape input -----
        self.layers['input'] = tf.reshape(self.input, [-1, 28, 28, 1])

        # ----- Stack 1 -----
        with tf.variable_scope('simple_net'):
            with tf.variable_scope('stack1'):
                # --- Convolution ---
                self.layers['st1_conv'] = tf.layers.conv2d(inputs      = self.layers['input'],
                                                           filters     = 16,
                                                           kernel_size = [3, 3],
                                                           strides     = [1, 1],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st1_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st1_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st1_relu'] = tf.nn.relu(features = self.layers['st1_batchnorm'],
                                                     name     = 'relu')

            # ----- Stack 2 -----
            with tf.variable_scope('stack2'):
                # --- Convolution ---
                self.layers['st2_conv'] = tf.layers.conv2d(inputs      = self.layers['st1_relu'],
                                                           filters     = 32,
                                                           kernel_size = [5, 5],
                                                           strides     = [2, 2],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st2_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st2_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st2_relu']      = tf.nn.relu(features = self.layers['st2_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 3 -----
            with tf.variable_scope('stack3'):
                # --- Convolution ---
                self.layers['st3_conv'] = tf.layers.conv2d(inputs      = self.layers['st2_relu'],
                                                           filters     = 48,
                                                           kernel_size = [3, 3],
                                                           strides     = [1, 1],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st3_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st3_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st3_relu']      = tf.nn.relu(features = self.layers['st3_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 4 -----
            with tf.variable_scope('stack4'):
                # --- Convolution ---
                self.layers['st4_conv'] = tf.layers.conv2d(inputs      = self.layers['st3_relu'],
                                                           filters     = 64,
                                                           kernel_size = [5, 5],
                                                           strides     = [2, 2],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st4_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st4_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st4_relu']      = tf.nn.relu(features = self.layers['st4_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 5 -----
            with tf.variable_scope('stack5'):
                # --- Convolution ---
                self.layers['st5_conv'] = tf.layers.conv2d(inputs      = self.layers['st4_relu'],
                                                           filters     = 64,
                                                           kernel_size = [3, 3],
                                                           strides     = [1, 1],
                                                           padding     = 'same',
                                                           activation  = None,
                                                           name        = 'conv')

                # --- Batch normalization ---
                self.layers['st5_batchnorm'] = tf.layers.batch_normalization(inputs   = self.layers['st5_conv'],
                                                                             center   = True,
                                                                             scale    = True,
                                                                             name     = 'batchnorm',
                                                                             training = self.state)

                # --- Relu ---
                self.layers['st5_relu']      = tf.nn.relu(features = self.layers['st5_batchnorm'],
                                                          name     = 'relu')

            # ----- Stack 6 -----
            with tf.variable_scope('stack6'):
                # --- Reshape ---
                self.layers['st6_reshape'] = tf.reshape(self.layers['st5_relu'], [-1, 7 * 7 * 64])

                # --- Fully Connected Layer ---
                self.layers['st6_fc'] = tf.layers.dense(inputs     = self.layers['st6_reshape'],
                                                        units      = 128,
                                                        activation = tf.nn.relu)

                # --- Dropout Layer ---
                self.layers['st6_dropout'] = tf.layers.dropout(inputs   = self.layers['st6_fc'],
                                                               rate     = 0.5,
                                                               training = self.state)

            # ----- Stack 4 -----
            with tf.variable_scope('stack7'):
                # --- Fully Connected Layer ---
                self.layers['st7_fc'] = tf.layers.dense(inputs = self.layers['st6_dropout'],
                                                        units  = 10)

                self.layers['st7_prob'] = tf.nn.softmax(logits = self.layers['st7_fc'],
                                                        name   = 'softmax')

                self.layers['st7_pred'] = tf.argmax(input       = self.layers['st7_fc'],
                                                    axis        = 1,
                                                    output_type = tf.int32)

        # ----- Loss function -----
        # --- Train ---
        self.loss   = tf.reduce_mean(-tf.log(tf.gather_nd(self.layers['st7_prob'],
                                                          tf.transpose([tf.range(self.batch_size), self.output]))
                                            )
                                     )
        _adam_opti = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        _params    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'simple_net')
        _grads     = tf.gradients(self.loss, _params)

        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
                                       _grads         = _grads,
                                       _params        = _params)
        def _train_func(_session, _state, _learning_rate, _batch_size,
                        _batch_x, _batch_y):
            return _session.run([self.loss, self.optimizer.ratio, self.optimizer.train_opt],
                                feed_dict = {
                                    'state:0':         _state,
                                    'learning_rate:0': _learning_rate,
                                    'batch_size:0':    _batch_size,
                                    'input:0':         _batch_x,
                                    'output:0':        _batch_y,
                                })
        self.train_func = _train_func

        # --- Valid ---
        self.prec = tf.reduce_mean(tf.cast(tf.equal(self.output, self.layers['st7_pred']), tf.float32))
        def _valid_func(_session, _state,
                        _batch_x, _batch_y):
            return _session.run([self.prec],
                                feed_dict = {
                                    'state:0':  _state,
                                    'input:0':  _batch_x,
                                    'output:0': _batch_y,
                                })
        self.valid_func = _valid_func

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]
