"""
StackDenseModel using tensorflo/keras 2.0. This will allow to append multiple layers to the Feed forward model one at a time.
author: harsh
date: Nov 2020
"""

import tensorflow as tf


class StackDenseModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, activation, initializer, batch_size):
        super(StackDenseModel, self).__init__()

        self.initializer = initializer
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.stack_layers = [
            tf.keras.layers.Dense(self.input_dim,
                                  activation=self.activation,
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=None, name=f"dense_{self.input_dim}"),
            tf.keras.layers.Dense(self.output_dim,
                                  activation=self.activation,
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=None, name=f"dense_{self.output_dim}")
        ]

    def append_dense_block(self, dim_d, x):
        self.stack_layers.remove(self.stack_layers[-1])
        self.stack_layers.append(tf.keras.layers.Dense(dim_d,
                                                       activation=self.activation,
                                                       kernel_initializer=self.initializer,
                                                       kernel_regularizer=None, name=f"dense_append{dim_d}"))
        self.stack_layers.append(tf.keras.layers.Dense(self.output_dim,
                                                       activation=self.activation,
                                                       kernel_initializer=self.initializer,
                                                       kernel_regularizer=None, name=f"dense_append{self.output_dim}"))
        self.call(x)

    def model(self):
        x_ = tf.keras.Input(shape=self.input_dim)
        return tf.keras.Model(inputs=[x_], outputs=self.call(x_))

    def call(self, inputs, training=None, mask=None):
        y = inputs
        for h in self.stack_layers:
            y = h(y)
        return y


if __name__ == '__main__':
    new_model = StackDenseModel(2, 3, 'relu', None, 32)
    new_model.build(input_shape=(32, 10))
    print(new_model.model().summary())
