import logging
import math
import tensorflow as tf

from functools import reduce
from operator import mul


class ContextualParameterGenerator(object):
    def __init__(self, context_size, name, dtype, shape, initializer, dropout=0.5, use_batch_norm=False,
                 batch_norm_momentum=0.99, batch_norm_train_stats=False):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_train_stats = batch_norm_train_stats
        self.num_elements = reduce(mul, self.shape, 1)

        # Create the projection matrices.
        self.projections = []
        in_size = context_size[0]
        for i, n in enumerate(context_size[1:] + [self.num_elements]):
            self.projections.append(
                tf.get_variable(
                    name='%s/CPG/Projection%d' % (name, i),
                    dtype=tf.float32,
                    shape=[in_size, n],
                    initializer=initializer))
            in_size = n

    def generate(self, context, is_train):
        # Generate the parameter values.
        generated_value = context
        for i, projection in enumerate(self.projections[:-1]):
            generated_value = tf.matmul(generated_value, projection)
            if self.use_batch_norm:
                is_train_batch_norm = is_train if self.batch_norm_train_stats else False
                generated_value = tf.layers.batch_normalization(
                    generated_value, momentum=self.batch_norm_momentum, reuse=tf.AUTO_REUSE,
                    training=is_train_batch_norm, fused=True, name='%s/CPG/Projection%d/BatchNorm' % (self.name, i))
            generated_value = tf.nn.relu(generated_value)
            generated_value = tf.nn.dropout(
                generated_value, 1 - (self.dropout * tf.cast(is_train, tf.float32)))

        generated_value = tf.matmul(generated_value, self.projections[-1])

        # Reshape and cast to the requested type.
        generated_value = tf.reshape(generated_value, self.shape)
        generated_value = tf.cast(generated_value, self.dtype)

        return generated_value