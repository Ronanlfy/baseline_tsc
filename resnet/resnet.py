from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation, Concatenate, GlobalAveragePooling1D, SpatialDropout1D, Add
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization, InputLayer
from tensorflow_addons.layers import WeightNormalization
import inspect

class ResNet(Model):
    """Creates a Inception layer / module.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A inception layer.
        """

    def __init__(self,
                 nb_filters=[64,64,128],
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='relu',
                 dropout_rate=0.1,
                 nb_classes=3,
                 entry_shape=None,
                 **kwargs):

        # initialize parent class
        super(ResNet, self).__init__(**kwargs)

        self.conv_layers, self.bn_layers, self.res_conv, self.res_bn = [], [], [], []

        self.activation = activation
        self.nb_classes = nb_classes
        self.entry_shape = entry_shape

        for i, fil in enumerate(nb_filters):
            self.conv_layers.append(Conv1D(filters=fil, kernel_size=kernel_size, strides=strides, 
                        padding=padding, use_bias=False))
            self.bn_layers.append(BatchNormalization())

            if i % 3 == 2:
                self.res_conv.append(Conv1D(filters=fil, kernel_size=1, strides=strides, 
                        padding=padding, use_bias=False))
                self.res_bn.append(BatchNormalization())
        
        self.dense = Dense(nb_classes, activation='softmax')

    def call(self, input_tensor, training=False):

        x = input_tensor
        input_res = input_tensor

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.bn_layers[i](x, training=training)
            x = Activation(self.activation)(x)

            if i % 3 == 2:
                shortcut = self.res_conv[i//3](input_res)
                shortcut = self.res_bn[i//3](shortcut, training=training)
            
                x = Add()([x, shortcut])
                x = Activation(self.activation)(x)

                input_res = x

        full = GlobalAveragePooling1D()(x)

        out = self.dense(full)

        return out

