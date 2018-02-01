import tensorflow as tf
from collections import OrderedDict
from Layers.Optimizer import *
from Layers.RBMLayer import *

class RBMModel():
    def _free_energy(self,
                     _input,
                     _W,
                     _hbias,
                     _vbias):
        _vbias_energy = tf.tensordot(_input, _vbias, axes = [[1], [0]])
        _wx_b         = tf.tensordot(_input, _W, axes = [[1], [0]]) + _hbias
        _hbias_energy = tf.reduce_sum(tf.log(1 + tf.exp(_wx_b)), axis = 1)
        return - _vbias_energy - _hbias_energy

    def __init__(self):
        # ===== Create tensor variables to store input / output data =====
        self.input         = tf.placeholder(tf.float32, shape = [None, 784], name = 'input')
        self.batch_size    = tf.placeholder(tf.int32, shape = (), name = 'batch_size')
        self.state         = tf.placeholder(tf.bool, shape = (), name = 'state')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name = 'learning_rate')

        # ===== Create model =====
        # ----- Create net -----
        self.net_name = 'CNN for Feature Extraction'
        self.layers   = OrderedDict()

        # ----- Reshape input -----
        self.layers['input'] = tf.reshape(self.input, [-1, 784])

        # ----- Stack 1 -----
        with tf.variable_scope('RBM'):
            with tf.variable_scope('stack1'):
                # --- RBM layer ---
                self.layers['st1_rbm'] = RBMLayer(_input      = self.layers['input'],
                                                  _name       = 'RBM',
                                                  _num_hidden = 500,
                                                  _batch_size = self.batch_size,
                                                  _n_steps    = 1)

        # ----- Loss function -----
        _W              = self.layers['st1_rbm'].W
        _hbias          = self.layers['st1_rbm'].hbias
        _vbias          = self.layers['st1_rbm'].vbias
        _gen_sample     = self.layers['st1_rbm'].gen_samples
        _gen_sig_sample = self.layers['st1_rbm'].gen_sig_samples

        # --- Train ---
        _input_energy      = self._free_energy(_input = self.input,
                                               _W     = _W,
                                               _hbias = _hbias,
                                               _vbias = _vbias)
        _gen_sample_energy = self._free_energy(_input = _gen_sample,
                                               _W     = _W,
                                               _hbias = _hbias,
                                               _vbias = _vbias)
        _loss       = tf.reduce_mean(_input_energy) - tf.reduce_mean(_gen_sample_energy)
        _recon_loss = tf.reduce_sum(tf.square(self.input - _gen_sig_sample)) / tf.cast(self.batch_size, dtype = tf.float32)
        _adam_opti  = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        _params     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'RBM')
        _grads      = tf.gradients(_loss, _params)
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(_update_ops):
            self.optimizer = Optimizer(_optimizer_opt = _adam_opti,
                                       _grads         = _grads,
                                       _params        = _params)
        def _train_func(_session, _state, _learning_rate,
                        _batch_size, _batch_x):
            return _session.run([_loss, _recon_loss, self.optimizer.ratio, self.optimizer.train_opt],
                                feed_dict = {
                                    'state:0':         _state,
                                    'learning_rate:0': _learning_rate,
                                    'batch_size:0':    _batch_size,
                                    'input:0':         _batch_x,
                                })
        self.train_func = _train_func

        # --- Valid ---
        def _valid_func(_session, _state,
                        _batch_size, _batch_x):
            return _session.run([_recon_loss],
                                feed_dict = {
                                    'state:0':      _state,
                                    'batch_size:0': _batch_size,
                                    'input:0':      _batch_x,
                                })
        self.valid_func = _valid_func

        # --- Reconstructor ---
        def _recon_func(_session, _state,
                        _batch_size, _batch_x):
            return _session.run([_gen_sig_sample],
                                feed_dict = {
                                    'state:0': _state,
                                    'input:0': _batch_x,
                                    'batch_size:0': _batch_size,
                                })
        self.recon_func = _recon_func

        # --- Reconstructor ---
        def _filter_func(_session):
            return _session.run([_W])
        self.filter_func = _filter_func

    def get_layer(self,
                  _layer_name):
        return self.layers[_layer_name]
