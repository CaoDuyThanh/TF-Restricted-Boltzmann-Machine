import tensorflow as tf
from collections import OrderedDict
from Layers.Optimizer import *
from Layers.RBMLayer import *

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
        self.layers['input'] = tf.reshape(self.input, [-1, 734])

        # ----- Stack 1 -----
        with tf.variable_scope('RBM'):
            with tf.variable_scope('stack1'):
                # --- RBM layer ---
                self.layers['st1_rbm'] = RBMLayer(_input = self.layers['input'],
                                                  _name  = 'W')



        tf.layers.dense()
        # # ----- Loss function -----
        # # --- Train ---
        # self.loss   = tf.reduce_mean(-tf.log(tf.gather_nd(self.layers['st7_prob'],
        #                                                   tf.transpose([tf.range(self.batch_size), self.output]))
        #                                     )
        #                              )
        # _adam_opti = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        # _params    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'RBM')
        # _grads     = tf.gradients(self.loss, _params)
        #
        # _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(_update_ops):
        #     self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
        #                                _grads         = _grads,
        #                                _params        = _params)
        # def _train_func(_session, _state, _learning_rate, _batch_size,
        #                 _batch_x, _batch_y):
        #     return _session.run([self.loss, self.optimizer.ratio, self.optimizer.train_opt],
        #                         feed_dict = {
        #                             'state:0':         _state,
        #                             'learning_rate:0': _learning_rate,
        #                             'batch_size:0':    _batch_size,
        #                             'input:0':         _batch_x,
        #                             'output:0':        _batch_y,
        #                         })
        # self.train_func = _train_func
        #
        # # --- Valid ---
        # self.prec = tf.reduce_mean(tf.cast(tf.equal(self.output, self.layers['st7_pred']), tf.float32))
        # def _valid_func(_session, _state,
        #                 _batch_x, _batch_y):
        #     return _session.run([self.prec],
        #                         feed_dict = {
        #                             'state:0':  _state,
        #                             'input:0':  _batch_x,
        #                             'output:0': _batch_y,
        #                         })
        # self.valid_func = _valid_func


    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]
