"""
############################################
Neural Network model class for evaluation
############################################

Requires:
    - Python 3.6 or higher
    - Tensorflow 1.13 or higher
    - h5py
"""

import os, sys, json, gc
import logging, warnings, h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, regularizers, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, explained_variance_score, r2_score, mean_absolute_error, mean_squared_error


class NNmodel():

    def __init__(self, Num_classes=3, learning_rate=0.001, batch_size=32, decay=0, epochs=100, stateful_mode=False, 
                  model=None, regression=False, **kwargs):

        self.Num_classes = Num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay = decay
        self.epochs = epochs
        self.stateful_mode = stateful_mode
        self.regression = regression
        if model is None:
            self.model = Sequential()
        else:
            self.init_model(model)


    def init_model(self, model):
        # compile the model and init some attr of the object by extracting info from the model
        self.model = model
        self.Num_classes = int(self.model.layers[-1].output_shape[-1])
        if self.Num_classes == 1:
            self.regression = True
        self.model_compile()


    def validate_model_shape(self, x):

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        self._validate_input_shape(x)
        self._validate_output_shape()


    def reset_session(self):
        # try to clear training session to speed up
        session.close()
        k.clear_session()


    def model_compile(self):
        # give model different compile methods
        if self.regression:
            self.model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(lr=self.learning_rate,
                                                                          decay=self.decay), metrics=['mae'])
        else:
            if self.Num_classes == 2:
                self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.learning_rate,
                                                                 decay=self.decay), metrics=['accuracy'])
            else:
                self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=self.learning_rate,
                                                                          decay=self.decay), metrics=['accuracy'])

    def evaluate(self, data_list):
        '''
        Main evaluation function
        '''
        def _metric_compute(set_key, display_name, data=None, data_generator=None):
            if self.regression:
                self._compute_metric_regression(data, data_generator, set_key, display_name, metric_dict)
            else:
                self._compute_metric_classification(data, data_generator, set_key, display_name, metric_dict)

        output_dict = {}
        output_dict['parameters'] = self.model.count_params()

        metric_dict = {} 

        for da in data_list:

            # compute for training set
            _metric_compute(**da)

        output_dict['metric'] = metric_dict

        return output_dict
        

    def _compute_metric_classification(self, data, data_generator, set_key, display_name, metric_dict):
        '''
        Main evaluation function for classification task
        :return: metrics_tr, metrics_val, metrics_te, where metrics includes accuracy, micro-F1 score, and confusion matrix
        '''
            
        if self.stateful_mode:
            self.model.reset_states()

        if data:
            X, y = data
            if X is None:
                return 
            y_pred = np.argmax(self.model.predict(X, batch_size=self.batch_size), axis=-1)

        else:
            data_generator.__reset_index__()
            y_pred = np.argmax(self.model.predict(x=data_generator), axis=-1)
            with h5py.File(data_generator.data, "r") as f:
                y = f["y_{}".format(data_generator.dataset)][:]
            data_generator.on_epoch_end()
            
        if y.ndim > 1:
            if y.shape[1] > 1:
                # y has been one hot encoded, decode it 
                y = np.argmax(y, axis=-1)
            else:
                y = np.squeeze(y)
            
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        cm = self._compute_confusion_matrix(y, y_pred)
        
        metric_key = ['accuracy', 'f1_score', 'confusion_matrix']

        metric = {}
        metric["display_name"] = display_name
        metric[metric_key[0]] = accuracy
        metric[metric_key[1]] = f1
        metric[metric_key[2]] = cm.tolist()

        metric_dict[set_key] = metric


    def _compute_confusion_matrix(self, y, y_pred):
        # compute the confusion matrix by hand
        cm = np.zeros((self.Num_classes, self.Num_classes))
        for i in range(len(y)):
                cm[int(y[i]), int(y_pred[i])] += 1
        return cm


    def _compute_metric_regression(self, data, data_generator, set_key, display_name, metric_dict):
        '''
        Main evaluation function for classification task
        :return: metrics_tr, metrics_val, metrics_te, where metrics includes accuracy, micro-F1 score, and confusion matrix
        '''
            
        if self.stateful_mode:
            self.model.reset_states()

        if data:
            X, y = data
            y_pred = np.argmax(self.model.predict(X, batch_size=self.batch_size), axis=-1)

        else:
            data_generator.__reset_index__()
            y_pred = np.argmax(self.model.predict(x=data_generator), axis=-1)
            with h5py.File(data_generator.data, "r") as f:
                y = f["y_{}".format(data_generator.dataset)][:]
            data_generator.on_epoch_end()

        metric_key = ['mean_squared_error', 'mean_abs_error', 'variance', 'r2_score']

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        evs = explained_variance_score(y, y_pred)
        r2 = r2_score(y, y_pred)

        metric = {}
        metric["display_name"] = display_name
        metric[metric_key[0]] = mse
        metric[metric_key[1]] = mae
        metric[metric_key[2]] = evs
        metric[metric_key[3]] = r2

        metric_dict[set_key] = metric



    def predict(self, x):
        '''
        :param x: a numpy testing array
        :return: a numpy array of probability in shape [m, n], m: the number of testing samples; n: the number of classes
        '''
    
        return self.model.predict(x, batch_size=self.batch.size)
    
    
    def predict_class(self, x):
        '''
        :param x: a numpy testing array
        :return: a numpy array of labels in shape [m, 1], m is the number of testing samples
        '''
    
        return self.model.predict_classes(x, batch_size=self.batch.size)


    def get_config(self):
        '''
        :return: summary of the model
        '''

        return self.model.summary()


   

    def initialize(self):
        '''
        re-initialize a trained model

        :param model: given a model
        :return: a new model which has replaced the original CuDNNLSTM with LSTM
        '''
        session = K.get_session()

        new_model = Sequential()

        for i in range(len(self.model.layers)):
            if not self.model.layers[i].get_config()['name'].find('batch_normalization') != -1:
                for v in self.model.layers[i].__dict__:
                    v_arg = getattr(self.model.layers[i], v)
                    if hasattr(v_arg, 'initializer'):
                        initializer_method = getattr(v_arg, 'initializer')
                        initializer_method.run(session=session)
                        print('reinitializing layer {}.{}'.format(self.model.layers[i].name, v))

        print(new_model.summary())

        new_model.compile(loss=self.model.loss, optimizer=self.model.optimizer)

        return new_model




def label_encode(y_tr, y_val, y_te=None, Num_classes=None):
    '''
    One-hot encode labels

    :param y_tr: training label list
    :param y_val: training labeSl list
    :param y_te: training label list
    :return: one-hot encoded y_tr, y_val and y_te
    '''

    try:
        dim1 = y_tr.shape[1]
        if dim1 >= 2:
            return y_tr, y_val, y_te
        else:
            y_tr = np.squeeze(y_tr)
            y_val = np.squeeze(y_val)
            y_te = np.squeeze(y_te)
    except:
        pass
            
    print("#################### One hot encoding labels ####################")
    if Num_classes is None:
        Num_classes =  max(np.hstack((y_tr, y_val, y_te))) + 1

    if  Num_classes > 1:
        Num_classes_tr = len(list(set(y_tr)))
    
    # check the label distribution
    check_label_distribution(y_tr, y_val, y_te)

    y_tr = to_categorical(y_tr, num_classes=Num_classes)
    y_val = to_categorical(y_val, num_classes=Num_classes)    
    y_te = to_categorical(y_te, num_classes=Num_classes)

    return y_tr, y_val, y_te


def check_label_distribution(y_tr, y_val, y_te):

    critical_error = False
    Num_classes_tr = len(list(set(y_tr)))
    Num_classes_val = len(list(set(y_val)))
    Num_classes_te = len(list(set(y_te)))
