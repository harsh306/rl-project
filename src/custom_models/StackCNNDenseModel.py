"""
StackCNNDenseModel using tensorflo/keras 2.0. This will automatically add multiple layers to the CNN+Feed forward model.
author: harsh
date: Dec 2020
"""
import tensorflow as tf


class StackCNNDenseModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers, activation, initializer, batch_size):
        super(StackCNNDenseModel, self).__init__()
        self.initializer = initializer
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.stack_layers = [
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 3),
                                   input_shape=input_dim,
                                   activation=self.activation,
                                   name=f"conv_1"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=f"pool_1"),
            tf.keras.layers.Flatten()
        ]
        self.stack_layers.extend(
            [
                tf.keras.layers.Dense(self.hidden_dim,
                                      activation=self.activation,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=None, name=f"dense_{d}")
                for d in range(num_hidden_layers)
            ]
        )
        self.stack_layers.append(tf.keras.layers.Dense(self.output_dim,
                                                       activation=self.activation,
                                                       kernel_initializer=self.initializer,
                                                       kernel_regularizer=None,
                                                       name=f"dense_out_{self.num_hidden_layers + 1}"))

    def model(self):
        x_ = tf.keras.Input(shape=self.input_dim)
        return tf.keras.Model(inputs=[x_], outputs=self.call(x_))

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

    def call(self, inputs, training=None, mask=None):
        y = inputs
        for h in self.stack_layers:
            y = h(y)
        return y


if __name__ == '__main__':
    input_dim = (10, 10, 3)
    output_dim = 9
    hidden_dim = 5
    num_hidden_dense_layers = 3
    new_model = StackCNNDenseModel(input_dim, output_dim, hidden_dim,
                                   num_hidden_dense_layers, 'relu', None, 32)
    new_model.build(input_shape=(32, 10, 10, 3))
    print(new_model.model().summary())
