import tensorflow as tf

class RBMLayer():
    def __init__(self,
                 _input,
                 _name,
                 _num_hidden,
                 _batch_size,
                 _n_steps = 1):
        _input_shape = _input.shape

        with tf.variable_scope(_name):
            # --- W param ---
            self.W = tf.get_variable(name        = 'W',
                                     shape       = (_input_shape[1], _num_hidden),
                                     dtype       = tf.float32,
                                     initializer = tf.random_normal_initializer(stddev = 0.01))

            # --- hbias param ---
            self.hbias = tf.get_variable(name        = 'hbias',
                                         shape       = (_num_hidden),
                                         dtype       = tf.float32,
                                         initializer = tf.random_normal_initializer(stddev = 0.01))

            # --- vbias param ---
            self.vbias = tf.get_variable(name        = 'vbias',
                                         shape       = (_input_shape[1]),
                                         dtype       = tf.float32,
                                         initializer = tf.random_normal_initializer(0.01))

        self.params = [self.W, self.hbias, self.vbias]

        # --- Generate sample ---
        _pre_h0, _sig_h0, _h0_sample = self._sample_h_given_v(_v_sample = _input, _batch_size = _batch_size)
        _pre_vn, _sig_vn, _vn_sample, _pre_hn, _sig_hn, _hn_sample = [], [], [], [], [], []
        for _step in range(_n_steps):
            _pre_v, _sig_v, _v_sample, \
            _pre_h, _sig_h, _h_sample = self._gibbs_hvh(_batch_size, _h0_sample)
            _h0_sample = _h_sample

            _pre_vn.append(_pre_v)
            _sig_vn.append(_sig_v)
            _vn_sample.append(_v_sample)
            _pre_hn.append(_pre_h)
            _sig_hn.append(_h_sample)
            _hn_sample.append(_h_sample)

        self.output          = _sig_h0
        self.gen_samples     = _vn_sample[-1]
        self.gen_sig_samples = _sig_vn[-1]

    def _sample_h_given_v(self,
                          _batch_size,
                          _v_sample):
        _pre_h1    = tf.matmul(_v_sample, self.W) + self.hbias
        _sig_h1    = tf.sigmoid(_pre_h1)
        _h1_sample = tf.keras.backend.random_binomial(shape = (_batch_size, _sig_h1.shape[1]),
                                                      p     = _sig_h1,
                                                      dtype = tf.float32)
        return _pre_h1, _sig_h1, _h1_sample

    def _sample_v_given_h(self,
                          _batch_size,
                          _h_sample):
        _pre_v1    = tf.matmul(_h_sample, tf.transpose(self.W)) + self.vbias
        _sig_v1    = tf.sigmoid(_pre_v1)
        _v1_sample = tf.keras.backend.random_binomial(shape = (_batch_size, _sig_v1.shape[1]),
                                                      p     = _sig_v1,
                                                      dtype = tf.float32)
        return _pre_v1, _sig_v1, _v1_sample

    def _gibbs_hvh(self,
                   _batch_size,
                   _h0_sample):
        _pre_v0, _sig_v0, _v0_sample = self._sample_v_given_h(_h_sample = _h0_sample, _batch_size = _batch_size)
        _pre_h1, _sig_h1, _h1_sample = self._sample_h_given_v(_v_sample = _v0_sample, _batch_size = _batch_size)
        return [_pre_v0, _sig_v0, _v0_sample,
                _pre_h1, _sig_h1, _h1_sample]

        idx = [1,3,4]

        a[idx, idx]

    def _gibbs_vhv(self,
                   _batch_size,
                   _v0_sample):
        _pre_h0, _sig_h0, _h0_sample = self._sample_h_given_v(_v_sample = _v0_sample, _batch_size = _batch_size)
        _pre_v1, _sig_v1, _v1_sample = self._sample_v_given_h(_h_sample = _h0_sample, _batch_size = _batch_size)
        return [_pre_h0, _sig_h0, _h0_sample,
                _pre_v1, _sig_v1, _v1_sample]


