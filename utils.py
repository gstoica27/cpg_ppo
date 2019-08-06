import numpy as np
import os
import sys
import tensorflow as tf
import gym
from collections import defaultdict
import atari_constants
import box_constants
from cpg import ContextualParameterGenerator


def wrap_env(env_name, context_size=None):
    env_data = defaultdict(lambda: None)
    env_data['env'] = gym.make(env_name)
    # env_data['context_size'] = context_size

    env_data['is_atari'] = len(env_data['env'].observation_space.shape) != 1
    return env_data


def create_fc_network_params(input_dim, context_dim, fc_architecture, initializer,
                             name, cpg_network_shape=None, dropout=.5, use_batch_norm=True,
                             batch_norm_momentum=.1, batch_norm_train_stats=True):
    if initializer is None:
        initializer = tf.orthogonal_initializer(np.sqrt(2.0))

    network_weights = []
    network_bias = []
    for layer_idx, hidden_neurons in enumerate(fc_architecture):
        if cpg_network_shape is not None:
            fc_weights = ContextualParameterGenerator(
                context_size=[context_dim] + cpg_network_shape,
                name=name + f'_{layer_idx}_weights',
                dtype=tf.float32,
                shape=[input_dim, hidden_neurons],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                initializer=initializer,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_train_stats=batch_norm_train_stats)
            fc_bias = ContextualParameterGenerator(
                context_size=[context_dim] + cpg_network_shape,
                name=name + f'_{layer_idx}_bias',
                dtype=tf.float32,
                shape=[hidden_neurons],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                initializer=initializer,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_train_stats=batch_norm_train_stats)
        else:
            fc_weights = tf.get_variable(
                name=name + f'_{layer_idx}_weights', dtype=tf.float32,
                shape=[input_dim, hidden_neurons],
                initializer=initializer)
            fc_bias = tf.get_variable(name=name + f'_{layer_idx}_bias', dtype=tf.float32,
                                      shape=[hidden_neurons],
                                      initializer=tf.zeros_initializer())

        input_dim = hidden_neurons

        network_weights.append(fc_weights)
        network_bias.append(fc_bias)

    return {name: {'weights': network_weights, 'bias': network_bias}}


def create_cnn_network_params(context_dim, cnn_architecture, padding, initializer,
                             name, cpg_network_shape=None, dropout=.5, use_batch_norm=True,
                             batch_norm_momentum=.1, batch_norm_train_stats=True):
    if initializer is None:
        initializer = tf.orthogonal_initializer(np.sqrt(2.0))

    conv_filters = []
    conv_bias = []
    conv_strides = []
    input_depth = 3

    for layer_idx, (num_outputs, kernel_size, stride) in enumerate(cnn_architecture):
        if cpg_network_shape is not None:
            filter = ContextualParameterGenerator(
                context_size=[context_dim] + cpg_network_shape,
                name=name + f'_{layer_idx}_filter',
                dtype=tf.float32,
                shape=[kernel_size, kernel_size, input_depth, num_outputs],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                initializer=initializer,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_train_stats=batch_norm_train_stats)
            bias = ContextualParameterGenerator(
                context_size=[context_dim] + cpg_network_shape,
                name=name + f'_{layer_idx}_bias',
                dtype=tf.float32,
                shape=[num_outputs],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                initializer=initializer,
                batch_norm_momentum=batch_norm_momentum,
                batch_norm_train_stats=batch_norm_train_stats)
        else:
            filter = tf.get_variable(
                name=name + f'_{layer_idx}_filter',
                dtype=tf.float32,
                shape=[kernel_size,kernel_size, input_depth, num_outputs],
                initializer=initializer)
            bias = tf.get_variable(
                name=name + f'_{layer_idx}_bias',
                dtype=tf.float32,
                shape=[num_outputs],
                initializer=tf.zeros_initializer())

        conv_filters.append(filter)
        conv_bias.append(bias)
        conv_strides.append(stride)
        input_depth = num_outputs

    return {name: {'filters': conv_filters,
                   'bias': conv_bias,
                   'padding': padding,
                   'strides': conv_strides}}


def create_lstm_network_params(input_dim, hidden_dim, context_dim, initializer,
                               name, cpg_network_shape=None, dropout=.5, use_batch_norm=True,
                               batch_norm_momentum=.1, batch_norm_train_stats=True,
                               init_scale=1.0):
    if cpg_network_shape is not None:
        input_weights = ContextualParameterGenerator(
                    context_size=[context_dim] + cpg_network_shape,
                    name=name + '_input_weights',
                    dtype=tf.float32,
                    shape=[input_dim, 4 * hidden_dim],
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    initializer=initializer,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_train_stats=batch_norm_train_stats)
        hidden_weights = ContextualParameterGenerator(
                    context_size=[context_dim] + cpg_network_shape,
                    name=name + '_hidden_weights',
                    dtype=tf.float32,
                    shape=[hidden_dim, 4 * hidden_dim],
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    initializer=initializer,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_train_stats=batch_norm_train_stats)
        bias = ContextualParameterGenerator(
                    context_size=[context_dim] + cpg_network_shape,
                    name=name + '_bias',
                    dtype=tf.float32,
                    shape=[4 * hidden_dim],
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    initializer=initializer,
                    batch_norm_momentum=batch_norm_momentum,
                    batch_norm_train_stats=batch_norm_train_stats)
    else:
        input_weights = tf.get_variable(name + "_input_weights",
                                        [input_dim, hidden_dim * 4],
                                        initializer=tf.orthogonal_initializer(init_scale))
        hidden_weights = tf.get_variable(name + "_hidden_weights",
                                         [hidden_dim, hidden_dim * 4],
                                         initializer=tf.orthogonal_initializer(init_scale))
        bias = tf.get_variable(name + '_bias', [4 * hidden_dim],
                               initializer=tf.zeros_initializer())

    return {name: {'input_weights': input_weights,
                   'hidden_weights':hidden_weights,
                   'bias': bias}}


def fc_network(input, fc_params, gen_vector=None, is_train=False, activation=tf.nn.relu):
    weights = fc_params['weights']
    bias = fc_params['bias']

    output = input
    for layer_weights, layer_bias in zip(weights, bias):
        # if gen_vector is not None:
        if isinstance(layer_weights, ContextualParameterGenerator):
            layer_weights = layer_weights.generate(gen_vector, is_train)
            layer_bias = layer_bias.generate(gen_vector, is_train)

            output = tf.matmul(output[:, None, :], layer_weights)[:, 0, :] + layer_bias

        else:
            output = tf.matmul(input, layer_weights) + layer_bias

        if activation is not None:
            output = activation(output)

    return output


def cnn_network(input, cnn_params, gen_vector=None, is_train=False, activation=tf.nn.relu):
    filters = cnn_params['filters']
    bias = cnn_params['bias']
    padding = cnn_params['padding']
    strides = cnn_params['strides']

    conv_output = input
    for layer_idx, (layer_filter, layer_bias) in enumerate(zip(filters, bias)):

        # if gen_vector is not None:
        if isinstance(layer_filter, ContextualParameterGenerator):
            layer_filter = layer_filter.generate(gen_vector, is_train)
            layer_bias = layer_bias.generate(gen_vector, is_train)


            conv_output = tf.nn.conv2d(
                input=conv_output, filter=layer_filter,
                strides=strides[layer_idx], padding=padding)

            # conv_output = conv((conv_output, layer_filter))
            # conv_output = tf.map_fn(fn=conv, elems=(conv_output, layer_filter))[0]
            conv_output = conv_output + layer_bias[None, None, None, :]

        else:
            conv_output = tf.nn.conv2d(
                input=conv_output, filter=layer_filter,
                strides=strides[layer_idx], padding='VALID')
            conv_output = conv_output + layer_bias[None, None, None, :]
        if activation is not None:
            conv_output = activation(conv_output)

    return conv_output


def lstm_network(inputs, keep_props, state, lstm_params, gen_vector=None, is_train=False):
    input_weights = lstm_params['input_weights']
    hidden_weights = lstm_params['hidden_weights']
    bias = lstm_params['bias']

    cell_state, hidden_state = tf.split(axis=1, num_or_size_splits=2, value=state)

    # if gen_vector is not None:
    if isinstance(input_weights, ContextualParameterGenerator):
        input_weights = input_weights.generate(gen_vector, is_train)
        hidden_weights = hidden_weights.generate(gen_vector, is_train)
        bias = bias.generate(gen_vector, is_train)

    for idx, (input, keep_prop) in enumerate(zip(inputs, keep_props)):
        cell_state = cell_state * (1 - keep_prop)
        hidden_state = hidden_state * (1 - keep_prop)

        if gen_vector is not None:
            input_transformed = tf.matmul(input[:, None, :], input_weights)[:, 0, :]
            hidden_transformed = tf.matmul(hidden_state[:, None, :], hidden_weights)[:, 0, :]

            layer_output = input_transformed + hidden_transformed + bias

        else:
            input_transformed = tf.matmul(input, input_weights)
            hidden_transformed = tf.matmul(hidden_state, hidden_weights)

            layer_output = input_transformed + hidden_transformed + bias

        input_gate, forget_gate, output_gate, union_gate = tf.split(axis=1,
                                                                    num_or_size_splits=4,
                                                                    value=layer_output)
        input_gate = tf.nn.sigmoid(input_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        output_gate = tf.nn.sigmoid(output_gate)
        union_gate = tf.tanh(union_gate)

        cell_state = forget_gate * cell_state + input_gate * union_gate
        hidden_state = output_gate * tf.tanh(cell_state)

        inputs[idx] = hidden_state
    state = tf.concat(axis=1, values=[cell_state, hidden_state])

    return inputs, state


