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
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda, MaxPooling1D, Concatenate, Add, GlobalAveragePooling1D
from tensorflow.keras.layers import Layer, Conv1D, Dropout, Dense, BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras import callbacks
from inception import inception

import sys, os, datetime
os.chdir(os.path.split(os.path.realpath(__file__))[0])
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir) 

from nnmodel import NNmodel
from data_generator import data_generator

data_h5 = "../data.hdf5"
Num_classes = 6
batch_size = 64

def build_model(num_feat,  # type: int
                data_len,  # type: int
                nb_filters,  # type: int
                strides=1,  # type: int
                nb_stacks=1, # type: int
                kernel_size=[10,20,40], # type: int
                padding='same',  # type: str
                dropout_rate=0.2,  # type: float
                activation='relu',
                name="inception",
                lr=0.001):

    def shortcut_layer(inputs, inception_out):

        shortcut_y = Conv1D(filters=inception_out.shape[-1], kernel_size=1, strides=strides,
                             padding=padding, use_bias=False)(inputs)
        shortcut_y = BatchNormalization()(shortcut_y)

        y = Add()([shortcut_y, inception_out])
        y = Activation(activation)(y)

        return y

    input_layer = Input(shape=(data_len, num_feat))
    input_res = input_layer
    x = input_layer

    for i in range(nb_stacks):
        x = inception(nb_filters=nb_filters, strides=strides, kernel_size=kernel_size,
                    padding=padding, activation=activation, dropout_rate=dropout_rate, name=name + "_{}".format(i+1))(x)

        if i % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
    
    x = GlobalAveragePooling1D()(x)

    print('x.shape=', x.shape)

    # x = Dropout(dropout_rate)(x)

    x = Dense(Num_classes, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    output_layer = x
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model
        
def run_task():
    
    training_generator, validation_generator, test_generator = data_generator(data_h5, Num_classes, batch_size)

    output_path = "models/model.h5"

    model = build_model(num_feat=33,
                        data_len=39,
                        nb_filters=3,
                        lr=0.001,
                        kernel_size=[20,30,40],
                        nb_stacks=3,
                        dropout_rate=0.12)

    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callback = [callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.0001), 
                callbacks.EarlyStopping(patience=50),
                callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    model.fit(x=training_generator, epochs=200, callbacks=callback,
              validation_data=validation_generator)
    
    model = load_model(output_path, custom_objects={'inception': inception})

    nmodel = NNmodel(model=model)
    data_list = [{"data_generator": training_generator, "set_key": "train_tf", "display_name": "TF Train"}, 
                    {"data_generator": validation_generator, "set_key": "validation_tf", "display_name": "TF Validation"},
                    {"data_generator": test_generator, "set_key": "test_tf", "display_name": "TF Test"}]

    metrics = nmodel.evaluate(data_list=data_list)

    print(metrics)

if __name__ == '__main__':

        run_task()