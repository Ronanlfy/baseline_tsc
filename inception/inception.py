from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Concatenate, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
from tensorflow_addons.layers import WeightNormalization
import inspect

class inception(Layer):
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
                 nb_filters=64,
                 strides=1,
                 padding='same',
                 activation='relu',
                 kernel_size=[10,20,40],
                 dropout_rate=0.1,
                 **kwargs):

        # initialize parent class
        super(inception, self).__init__(**kwargs)

        self.nb_filters = nb_filters
        self.activation = activation
        self.strides = strides
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate

    def _build_conv(self, input_shape, output_shape=[]):
        
        for i, kernel in enumerate(self.kernel_size):

            tmp_layer = []
            name_conv = 'conv1D_{}'.format(i+1)
            with K.name_scope(name_conv):
                tmp_layer.append(Conv1D(filters=self.nb_filters, kernel_size=kernel, strides=self.strides, 
                        padding=self.padding, use_bias=False, activation=self.activation, name=name_conv))
                tmp_layer[-1].build(input_shape)
                output_shape_conv = tmp_layer[-1].compute_output_shape(input_shape)
            
            name_bn = 'batchNorm_{}'.format(i+1)
            with K.name_scope(name_bn):
                tmp_layer.append(BatchNormalization(name=name_bn))
                tmp_layer[-1].build(output_shape_conv)

            self.conv_layers.append(tmp_layer)
            output_shape.append(tmp_layer[-1].compute_output_shape(output_shape_conv))

    def _build_residual(self, input_shape):

        self.residual_layers.append(MaxPooling1D(pool_size=3, strides=self.strides, padding=self.padding))
        self.residual_layers[-1].build(input_shape)
        output_shape = self.residual_layers[-1].compute_output_shape(input_shape)

        name = 'conv1D_bottleneck_1'
        with K.name_scope(name):
            self.residual_layers.append(Conv1D(filters=self.nb_filters, kernel_size=1, strides=self.strides, 
                                        padding=self.padding, use_bias=False, activation=self.activation, name=name))
            self.residual_layers[-1].build(output_shape)
            output_shape = self.residual_layers[-1].compute_output_shape(output_shape)

        return output_shape


    def build(self, input_shape):
        
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            
            self.conv_layers = []
            self.residual_layers = []

            name = 'conv1D_bottleneck_0'
            with K.name_scope(name):  # name scope used to make sure weights get unique names
                self.bot_layers = Conv1D(filters=self.nb_filters, kernel_size=1, strides=self.strides, 
                        padding=self.padding, use_bias=False, name=name)
                self.bot_layers.build(input_shape)    
                bot_output = self.bot_layers.compute_output_shape(input_shape) 

            conv_output_shape = []

            self._build_conv(bot_output, output_shape=conv_output_shape)  
            
            output_shape = self._build_residual(input_shape)

            conv_output_shape.append(output_shape)

            concate_layer = Concatenate(axis=2)
            concate_layer.build(conv_output_shape)
            output_shape = concate_layer.compute_output_shape(conv_output_shape) 

            with K.name_scope('norm_0'):
                self.batchnorm = BatchNormalization()
                self.batchnorm.build(output_shape)

        super(inception, self).build(input_shape)

    def call(self, inputs, training=False):

        training_flag = 'training' in dict(inspect.signature(self.bot_layers.call).parameters)
        bot = layer(inputs, training=training) if training_flag else self.bot_layers(inputs)

        conv_layer_output = []

        for layers in self.conv_layers:
            # layers is a list [conv, batch norm]
            for i, layer in enumerate(layers):
                training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
                if i == 0:
                    x_tmp = layer(bot, training=training) if training_flag else layer(bot)
                else:
                    x_tmp = layer(x_tmp, training=training) if training_flag else layer(x_tmp)
                    x_tmp = Activation(self.activation)(x_tmp)
                    x_tmp = SpatialDropout1D(rate=self.dropout_rate)(x_tmp)

                    conv_layer_output.append(x_tmp)

        x = inputs
        for layer in self.residual_layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
        
        conv_layer_output.append(x)

        x = Concatenate(axis=2)(conv_layer_output)
        x = self.batchnorm(x)
        x = Activation(activation=self.activation)(x)

        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(inception, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['strides'] = self.strides
        config['kernel_size'] = self.kernel_size
        config['padding'] = self.padding
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        return config

