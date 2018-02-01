# Load system libs
import timeit
from random import shuffle

# Import Models
from Models.RBMModel import *
from Models.VisualBackPropModel import *

# Import Utils
from Utils.MNistDataHelper import *      # Load dataset
from matplotlib import pyplot as plt     # Plot result
from Utils.TSBoardHandler import *       # Tensorboard handler

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# TRAINING HYPER PARAMETER
TRAIN_STATE         = True     # Training state
VALID_STATE         = False    # Validation state
BATCH_SIZE          = 16
NUM_EPOCH           = 50
LEARNING_RATE       = 0.003        # Starting learning rate
DISPLAY_FREQUENCY   = 500;         INFO_DISPLAY = '\r%sLearning rate = %f - Epoch = %d - Iter = %d - Cost = %f - Recon cost = %f'
SAVE_FREQUENCY      = 2500
VALIDATE_FREQUENCY  = 2500
VISUALIZE_FREQUENCY = 2500

START_EPOCH     = 0
START_ITERATION = 0

# EARLY STOPPING
PATIENCE              = 500000
PATIENCE_INCREASE     = 2
IMRROVEMENT_THRESHOLD = 0.995

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/Projects/Machine Learning/Dataset/MNIST/mnist.pkl.gz'

# STATE PATH
SETTING_PATH      = '../Pretrained/'
TSBOARD_PATH      = SETTING_PATH + 'Tensorboard/'
RECORD_PATH       = SETTING_PATH + 'RBM_Record.ckpt'
STATE_PATH        = SETTING_PATH + 'RBM_CurrentState.ckpt'
BEST_PREC_PATH    = SETTING_PATH + 'RBM_Prec_Best.ckpt'

#  GLOBAL VARIABLES
dataset      = None
RBM_model    = None
TB_hanlder   = None

########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def _load_dataset(_all_path):
    global dataset

    dataset = MNistDataHelper(_dataset_path = _all_path['dataset_path'])
    print ('|-- Load path = %s ! Completed !' % (_all_path['dataset_path']))

########################################################################################################################
#                                                                                                                      #
#    CREATE FEATURE EXTRACTION MODEL                                                                                   #
#                                                                                                                      #
########################################################################################################################
def _create_RBM_model():
    global RBM_model
    RBM_model = RBMModel()

########################################################################################################################
#                                                                                                                      #
#    CREATE TENSOR BOARD                                                                                               #
#                                                                                                                      #
########################################################################################################################
def _create_TSBoard_model(_all_path):
    global RBM_model, \
           TB_hanlder
    TB_hanlder = TSBoardHandler(_save_path = _all_path['tsboard_path'])

########################################################################################################################
#                                                                                                                      #
#    UTILITIES ..........                                                                                              #
#                                                                                                                      #
########################################################################################################################
def _split_data(_data,
                _ratios):
    splitted_data = []
    _num_samples  = len(_data)
    for _idx in range(len(_ratios) - 1):
        splitted_data.append(_data[int(_ratios[_idx]     * _num_samples) :
                                   int(_ratios[_idx + 1] * _num_samples), ])
    return splitted_data

def _sort(_data,
          _label):
    _idx = numpy.argsort(_label)
    sorted_data  = _data[_idx, ]
    sorted_label = _label[_idx, ]
    return sorted_data, sorted_label

def _shuffle_data(_data):
    _idx = range(len(_data[0]))
    numpy.random.RandomState().shuffle(_idx)
    data = [_batch[_idx,]
            for _batch in _data]
    return data

def _extract_feature(_model,
                     _set_x):
    _num_sample = len(_set_x)
    _num_batch  = int(numpy.ceil(_num_sample * 1.0 / BATCH_SIZE))
    feature = []
    for _batch_id in range(_num_batch):
        _sub_set_x = _set_x[_batch_id      * BATCH_SIZE :
                           (_batch_id + 1) * BATCH_SIZE, ]
        _result = _model.feat_ext_func(VALID_STATE,
                                       VALID_STATE,
                                       VALID_STATE,
                                       len(_sub_set_x),
                                       _sub_set_x)
        feature.append(_result[0])
    feature = numpy.concatenate(tuple(feature), axis = 0)
    return feature

def _scale_linear(_data,
                 _min,
                 _max):
    _min_data = numpy.min(_data)
    _max_data = numpy.max(_data)
    return (_data - _min_data) / (_max_data - _min_data) * (_max - _min) + _min

def _scale_log(_data,
               _min,
               _max):
    _data = numpy.log(_data)
    return _scale_linear(_data, _min, _max)

def _flat_to_imgs(_data):
    _num_imgs   = len(_data)
    _row        = int(numpy.sqrt(_num_imgs))
    _img_size   = _data.shape[1]
    _count      = 0
    merged_imgs = numpy.zeros((_row * _img_size, _row * _img_size))
    for _i in range(_row):
        for _j in range(_row):
            merged_imgs[_i * _img_size : (_i + 1) * _img_size,
                        _j * _img_size : (_j + 1) * _img_size] = _data[_count, :]
            _count += 1
    return merged_imgs

########################################################################################################################
#                                                                                                                      #
#    VALID FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
import scipy
def _valid_model(_session,
                 _model,
                 _valid_data):
    _valid_set_x, _valid_set_y = _valid_data
    _valid_set_x, \
    _valid_set_y = _shuffle_data([_valid_set_x, _valid_set_y])

    # --- Info params ---
    _loss = []
    _iter = 0
    _num_batch_valided_data = len(_valid_set_x) // BATCH_SIZE
    for _id_batch_valided_data in range(_num_batch_valided_data):
        _valid_start_time = timeit.default_timer()

        _valid_batch_x = _valid_set_x[ _id_batch_valided_data      * BATCH_SIZE:
                                      (_id_batch_valided_data + 1) * BATCH_SIZE, ]
        _valid_batch_y = _valid_set_y[ _id_batch_valided_data      * BATCH_SIZE:
                                      (_id_batch_valided_data + 1) * BATCH_SIZE, ]
        _iter += 1
        _result = _model.valid_func(_session    = _session,
                                    _state      = VALID_STATE,
                                    _batch_size = BATCH_SIZE,
                                    _batch_x    = _valid_batch_x)
        _loss.append(_result[0])
        _valid_end_time = timeit.default_timer()
        # Print information
        print '\r|-- Valid %d / %d batch - Time = %f' % (_id_batch_valided_data, _num_batch_valided_data, _valid_end_time - _valid_start_time),

        if _iter % DISPLAY_FREQUENCY == 0:
            # Print information of current training in progress
            print ('Loss = %f' % (numpy.mean(_loss)))

    return numpy.mean(_loss)

########################################################################################################################
#                                                                                                                      #
#    TRAIN FEATURE EXTRACTION MODEL..........                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_test_model(_all_path):
    global dataset, \
           RBM_model, \
           TB_hanlder
    # ===== Prepare path =====
    _setting_path     = _all_path['setting_path']
    _record_path      = _all_path['record_path']
    _state_path       = _all_path['state_path']
    _best_prec_path   = _all_path['best_prec_path']

    # ===== Prepare dataset =====
    _train_set_x = dataset.train_set_x
    _train_set_y = dataset.train_set_y
    _train_set_x, \
    _train_set_y = _shuffle_data([_train_set_x, _train_set_y])

    _visual_set_x = _train_set_x[:25]
    _visual_set_y = _train_set_y[:25]

    _valid_set_x = dataset.valid_set_x
    _valid_set_y = dataset.valid_set_y
    _valid_set_x, \
    _valid_set_y = _shuffle_data([_valid_set_x, _valid_set_y])

    _test_set_x = dataset.test_set_x
    _test_set_y = dataset.test_set_y
    _test_set_x, \
    _test_set_y = _shuffle_data([_test_set_x, _test_set_y])

    # ===== Start session =====
    _config  = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _session = tf.Session(config = _config)

    # ----- Initialize params -----
    _session.run(tf.global_variables_initializer())
    _session.run(tf.local_variables_initializer())

    # ----- Save graph -----
    TB_hanlder.save_graph(_graph = _session.graph)

    # ===== Load data record =====
    print ('|-- Load previous record !')
    iter_train_record       = []
    cost_train_record       = []
    recon_cost_train_record = []
    iter_valid_record       = []
    prec_valid_record       = []
    best_loss_valid         = tf.float32.max
    _epoch  = START_EPOCH
    _iter   = START_ITERATION
    if check_file_exist(_record_path, _throw_error = False):
        _file = open(_record_path, 'rb')
        iter_train_record       = pickle.load(_file)
        cost_train_record       = pickle.load(_file)
        recon_cost_train_record = pickle.load(_file)
        iter_valid_record       = pickle.load(_file)
        prec_valid_record       = pickle.load(_file)
        best_loss_valid         = pickle.load(_file)
        _epoch                  = pickle.load(_file)
        _iter                   = pickle.load(_file)
        _file.close()
    print ('|-- Load previous record ! Completed !')

    # ===== Load state =====
    _saver = tf.train.Saver()
    print ('|-- Load state !')
    if tf.train.checkpoint_exists(_state_path):
        _saver.restore(sess      = _session,
                       save_path = _state_path)
    print ('|-- Load state ! Completed !')

    # ===== Training start =====
    # ----- Temporary record -----
    _cost_train_temp       = []
    _recon_cost_train_temp = []
    _ratios                = []
    _learning_rate         = LEARNING_RATE

    # ----- Train -----
    while (_epoch < NUM_EPOCH):
        _epoch += 1

        # --- Train triplets ---
        _train_set_x, \
        _train_set_y = _shuffle_data([_train_set_x, _train_set_y])
        _num_batch_trained_data   = len(_train_set_x) / (BATCH_SIZE)
        for _id_batch in range(_num_batch_trained_data):
            _iter += 1

            _train_start_time = timeit.default_timer()
            _train_batch_x = _train_set_x[_id_batch      * BATCH_SIZE :
                                         (_id_batch + 1) * BATCH_SIZE, ]
            _train_batch_y = _train_set_y[_id_batch      * BATCH_SIZE :
                                         (_id_batch + 1) * BATCH_SIZE, ]

            _train_result = RBM_model.train_func(_session       = _session,
                                                 _state         = TRAIN_STATE,
                                                 _learning_rate = _learning_rate,
                                                 _batch_size    = BATCH_SIZE,
                                                 _batch_x       = _train_batch_x)

            # Temporary save info
            _cost_train_temp.append(_train_result[0])
            _recon_cost_train_temp.append(_train_result[1])
            _ratios.append(_train_result[2])
            _train_end_time = timeit.default_timer()

            # Print information
            print '\r|-- Trained %d / %d batch - Time = %f' % (_id_batch, _num_batch_trained_data, _train_end_time - _train_start_time),

            if _iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % ('|-- ', _learning_rate, _epoch, _iter, numpy.mean(_cost_train_temp), numpy.mean(_recon_cost_train_temp)))
                iter_train_record.append(_iter)
                cost_train_record.append(numpy.mean(_cost_train_temp))
                recon_cost_train_record.append(numpy.mean(_recon_cost_train_temp))
                print ('|-- Ratio = %f' % (numpy.mean(_ratios)))

                # Add summary
                TB_hanlder.log_scalar(_name_scope = 'Metadata',
                                      _name       = 'Learning rate',
                                      _value      = _learning_rate,
                                      _step       = _iter)
                TB_hanlder.log_scalar(_name_scope = 'Train',
                                      _name       = 'Loss',
                                      _value      = numpy.mean(_cost_train_temp),
                                      _step       = _iter)
                TB_hanlder.log_scalar(_name_scope = 'Train',
                                      _name       = 'Recon Loss',
                                      _value      = numpy.mean(_recon_cost_train_temp),
                                      _step       = _iter)
                TB_hanlder.log_scalar(_name_scope = 'Train',
                                      _name       = 'Ratio',
                                      _value      = numpy.mean(_ratios),
                                      _step       = _iter)

                # Reset list
                _cost_train_temp  = []
                _ratios = []

            if _iter % SAVE_FREQUENCY == 0:
                # Save record
                _file = open(_record_path, 'wb')
                pickle.dump(iter_train_record, _file, 2)
                pickle.dump(cost_train_record, _file, 2)
                pickle.dump(recon_cost_train_record, _file, 2)
                pickle.dump(iter_valid_record, _file, 2)
                pickle.dump(prec_valid_record, _file, 2)
                pickle.dump(best_loss_valid, _file, 2)
                pickle.dump(_epoch, _file, 2)
                pickle.dump(_iter, _file, 2)
                _file.close()
                print ('+ Save record ! Completed !')

                # Save state
                _saver.save(sess      = _session,
                            save_path = _state_path)
                print ('+ Save state ! Completed !')

            if _iter % VALIDATE_FREQUENCY == 0:
                print ('\n------------------- Validate Model -------------------')
                _prec_valid = _valid_model(_session    = _session,
                                           _model      = RBM_model,
                                           _valid_data = [_valid_set_x, _valid_set_y])
                iter_valid_record.append(_iter)
                prec_valid_record.append(_prec_valid)
                print ('\n+ Validate model finished! Loss = %f' % (_prec_valid))
                print ('\n------------------- Validate Model (Done) -------------------')

                # Add summary
                TB_hanlder.log_scalar(_name_scope = 'Valid',
                                      _name       = 'Loss',
                                      _value      = _prec_valid,
                                      _step       = _iter)

                # Save model if its cost better than old one
                if (_prec_valid < best_loss_valid):
                    best_loss_valid = _prec_valid

                    # Save best model
                    _saver.save(sess      = _session,
                                save_path = _best_prec_path)
                    print ('+ Save best loss model ! Complete !')

            if _iter % VISUALIZE_FREQUENCY == 0:
                print ('\n------------------- Visualize Model -------------------')
                _recon_imgs = RBM_model.recon_func(_session    = _session,
                                                   _state      = TRAIN_STATE,
                                                   _batch_size = len(_visual_set_x),
                                                   _batch_x    = _visual_set_x)[0]
                _recon_imgs   = _recon_imgs.reshape((len(_recon_imgs), 28, 28))
                _recon_imgs   = _scale_linear(_recon_imgs, 0, 255)
                _recon_imgs   = _flat_to_imgs(_recon_imgs)
                _origin_set_x = _visual_set_x.reshape((len(_visual_set_x), 28, 28))
                _origin_set_x = _scale_linear(_origin_set_x, 0, 255)
                _origin_set_x = _flat_to_imgs(_origin_set_x)
                _merged_imgs  = numpy.concatenate((_recon_imgs, _origin_set_x), axis = 1)
                # Add summary
                TB_hanlder.log_images(_name_scope = 'Train',
                                      _name       = 'Merged images',
                                      _images     = [_merged_imgs],
                                      _step       = _iter)

                _filters = RBM_model.filter_func(_session = _session)[0]
                _filters = numpy.transpose(_filters, (1, 0))
                _filters = _filters.reshape((len(_filters), 28, 28))
                _filters = _scale_linear(_filters, 0, 255)
                _filters = _flat_to_imgs(_filters)
                # Add summary
                TB_hanlder.log_images(_name_scope = 'Train',
                                      _name       = 'Filters',
                                      _images     = [_filters],
                                      _step       = _iter)
                print ('\n------------------- Visualize Model (Done) ------------')

    # ===== Load best model =====
    _saver = tf.train.Saver()
    print ('|-- Load best model !')
    if tf.train.checkpoint_exists(_best_prec_path):
        _saver.restore(sess      = _session,
                       save_path = _best_prec_path)
    print ('|-- Load best model ! Completed !')

    # ===== Testing =====
    print ('\n------------------- Test Model -------------------')
    _prec_test = _valid_model(_session    = _session,
                              _model      = RBM_model,
                              _valid_data = [_test_set_x, _test_set_y])
    print ('\n+ Test model finished! Prec = %f' % (_prec_test))
    print ('\n------------------- Test Model (Done) -------------------')

    # ===== Close session =====
    _session.close()

if __name__ == '__main__':
    _all_path                     = dict()
    _all_path['dataset_path']     = DATASET_PATH
    _all_path['setting_path']     = SETTING_PATH;     check_path_and_create(_all_path['setting_path'])
    _all_path['tsboard_path']     = TSBOARD_PATH;     check_path_and_create(_all_path['tsboard_path'])
    _all_path['record_path']      = RECORD_PATH
    _all_path['state_path']       = STATE_PATH
    _all_path['best_prec_path']   = BEST_PREC_PATH

    _load_dataset(_all_path = _all_path)
    _create_RBM_model()
    _create_TSBoard_model(_all_path = _all_path)
    _train_test_model(_all_path = _all_path)
