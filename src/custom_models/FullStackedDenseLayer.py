"""
FullStackedDenseModel using tensorflo/keras 2.0. This will automatically add multiple layers to the Feed forward model.
author: harsh
date: Nov 2020
"""
import tensorflow as tf


class FullStackedDenseModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation, initializer, batch_size):
        super(FullStackedDenseModel, self).__init__()
        self.initializer = initializer
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.stack_layers = [
            tf.keras.layers.Dense(self.hidden_dim,
                                  activation=self.activation,
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=None, name=f"dense_{d}")
            for d in range(num_hidden_layers)
        ]
        self.stack_layers.append(tf.keras.layers.Dense(self.output_dim,
                                                       activation=self.activation,
                                                       kernel_initializer=self.initializer,
                                                       kernel_regularizer=None,
                                                       name=f"dense_{self.num_hidden_layers + 1}"))

    def model(self):
        x_ = tf.keras.Input(shape=self.input_dim)
        return tf.keras.Model(inputs=[x_], outputs=self.call(x_))

    def call(self, inputs, training=None, mask=None):
        y = inputs
        for h in self.stack_layers:
            y = h(y)
        return y


if __name__ == '__main__':
    new_model = FullStackedDenseModel(4, 4, 5, 3, 'relu', None, 32)
    new_model.build(input_shape=(32, 4))
    print(new_model.model().summary())
