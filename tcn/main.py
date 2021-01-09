# set up tensorflow training config
import tensorflow as tf
##################################
# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of 10% the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

session = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
###################################


from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda, MaxPooling1D
from tensorflow.keras.layers import Layer, Conv1D, Dropout, Dense, BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras import callbacks
from tcn import TCN

import sys, os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir) 

from nnmodel import NNmodel
from data_generator import data_generator

data_h5 = "../data.hdf5"
Num_classes = 6
batch_size = 64


def build_model(num_feat,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=False,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.2,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='relu',
                 lr=0.001,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False):

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm, use_weight_norm,
            name=name)(input_layer)

    print('x.shape=', x.shape)

    x = MaxPooling1D(3)(x)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(2 * Num_classes, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(Num_classes, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    output_layer = x
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model
        
def run_task():
    
    training_generator, validation_generator, test_generator = data_generator(data_h5)

    output_path = "./model.h5"

    model = build_model(return_sequences=True,
                         num_feat=33,
                         nb_filters=12,
                         kernel_size=3,
                         dilations=[2 ** i for i in range(6)],
                         nb_stacks=1,
                         max_len=39,
                         lr=0.001,
                         dropout_rate=0.2, 
                         use_layer_norm=True,
                         use_batch_norm=False,
                         use_skip_connections=True)

    model.summary()

    callback = [callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=20, min_lr=0.0001), 
                callbacks.EarlyStopping(patience=20)]

    model.fit(x=training_generator, epochs=200, callbacks=callback,
              validation_data=validation_generator)
    
    model = load_model(output_path, custom_objects={'TCN': TCN})

    nmodel = NNmodel(model=model)
    data_list = [{"data_generator": training_generator, "set_key": "train_tf", "display_name": "TF Train"}, 
                    {"data_generator": validation_generator, "set_key": "validation_tf", "display_name": "TF Validation"},
                    {"data_generator": test_generator, "set_key": "test_tf", "display_name": "TF Test"}]

    metrics = nmodel.evaluate(data_list=data_list)

    print(metrics)

if __name__ == '__main__':

        run_task()