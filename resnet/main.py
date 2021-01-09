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
from resnet import ResNet

import sys, os, datetime
os.chdir(os.path.split(os.path.realpath(__file__))[0])
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir) 

from nnmodel import NNmodel
from data_generator import data_generator

data_h5 = "../data.hdf5"
Num_classes = 6
batch_size = 64

def build_model(input_shape=None,
                 nb_filters=[64,64,128],
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='relu',
                 dropout_rate=0.1,
                 name="inception",
                 lr=0.001):

    
    model = ResNet(nb_filters=nb_filters, nb_classes=Num_classes, activation=activation,
                        padding=padding, strides=strides, entry_shape=input_shape)
    
    model.build((None, *input_shape))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model
        
def run_task():
    
    training_generator, validation_generator, test_generator = data_generator(data_h5, Num_classes, batch_size)

    output_path = "models/model.h5"

    config = {"input_shape": (39,33),
                "nb_filters": [8,8,16],
                "lr":0.001,
                "kernel_size":6,
                "dropout_rate": 0.1}
    model = build_model(**config)

    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    callback = [callbacks.ModelCheckpoint(output_path, verbose=1, save_best_only=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.0001), 
                callbacks.EarlyStopping(patience=50),
                callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

    model.fit(x=training_generator, epochs=200, callbacks=callback,
              validation_data=validation_generator)

    # model = build_model(**config)
    # model.load_weights(output_path)

    nmodel = NNmodel(model=model)
    data_list = [{"data_generator": training_generator, "set_key": "train_tf", "display_name": "TF Train"}, 
                    {"data_generator": validation_generator, "set_key": "validation_tf", "display_name": "TF Validation"},
                    {"data_generator": test_generator, "set_key": "test_tf", "display_name": "TF Test"}]

    metrics = nmodel.evaluate(data_list=data_list)

    print(metrics)

if __name__ == '__main__':

        run_task()