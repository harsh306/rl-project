"""
SimpleStackedDenseModel using tensorflo/keras 2.0. This will automatically add multiple layers to the Feed forward model.
author: harsh
date: Nov 2020
"""
import tensorflow as tf


class SimpleStackedDenseModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, activation, initializer, batch_size):
        super(SimpleStackedDenseModel, self).__init__()
        self.initializer = initializer
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.stack_layers = [
            tf.keras.layers.Dense(d,
                                  activation=self.activation,
                                  kernel_initializer=self.initializer,
                                  kernel_regularizer=None, name=f"dense_{d}")
            for d in range(input_dim, output_dim - 1, -1)
        ]

    def model(self):
        x_ = tf.keras.Input(shape=self.input_dim)
        return tf.keras.Model(inputs=[x_], outputs=self.call(x_))

    def call(self, inputs, training=None, mask=None):
        y = inputs
        for h in self.stack_layers:
            y = h(y)
        return y


if __name__ == '__main__':
    new_model = SimpleStackedDenseModel(10, 2, 'relu', None, 32)
    new_model.build(input_shape=(32, 10))
    print(new_model.model().summary())
