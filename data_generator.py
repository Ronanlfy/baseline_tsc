import numpy as np
import tensorflow as tf
import h5py

class DataGenerator(tf.keras.utils.Sequence):
    """
    Batch data generator for tensorflow keras training

    """
    def __init__(self, data, dataset, batch_size=32, n_classes=10, shuffle=True):
        
        self.data = data
        self.dataset = dataset
        self.length = self.__get_data_length()
        self.shape = self.__get_data_shape()

        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __get_data_length(self):

        with h5py.File(self.data, "r") as f:
            length = len(f["y_{}".format(self.dataset)])
        return length

    def __get_data_shape(self):

        with h5py.File(self.data, "r") as f:
            shape = f["X_{}".format(self.dataset)][0].shape
        return shape

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        index_slice = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(index_slice)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_slice):
        'Generates keras data generator'

        index_slice.sort()

        with h5py.File(self.data, "r") as f:
        
            X = f["X_{}".format(self.dataset)][index_slice]
            y = f["y_{}".format(self.dataset)][index_slice]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


    def __reset_index__(self):
        'reset index'

        self.indexes = np.arange(self.length)


    def tf_generator(self):
        'Generates tf.data.dataset pipeline'

        self.indexes = np.arange(self.length)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        for index in range(self.__len__()):

            index_slice = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            index_slice.sort()

            with h5py.File(self.data, "r") as f:
        
                X = f["X_{}".format(self.dataset)][index_slice]
                y = f["y_{}".format(self.dataset)][index_slice]

            if (index == self.__len__() - 1) and len(index_slice) < self.batch_size:
                continue

            yield tf.convert_to_tensor(X), tf.keras.utils.to_categorical(y, num_classes=self.n_classes)



def data_generator(data_h5, Num_classes=6, batch_size=64):

    training_generator = DataGenerator(data=data_h5, dataset="train", n_classes=Num_classes, batch_size=batch_size)
    validation_generator = DataGenerator(data=data_h5, dataset="validation", n_classes=Num_classes, batch_size=batch_size)
    test_generator = DataGenerator(data=data_h5, dataset="test", n_classes=Num_classes, batch_size=batch_size)

    return training_generator, validation_generator, test_generator
