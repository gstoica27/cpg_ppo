import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from rlsaber.tf_util import lstm, batch_to_seq, seq_to_batch
from utils import *
from collections import defaultdict


class Network(object):
    def __init__(self,
                 convs,
                 fcs,
                 use_lstm=True,
                 padding='VALID',
                 continuous=False,
                 context_size=None,
                 cpg_network_shape=None,
                 dropout=.5,
                 use_batch_norm=True,
                 batch_norm_momentum=.1,
                 batch_norm_train_stats=True):

        self.all_variables = {}  # defaultdict(lambda: defaultdict(None))
        self.conv_architecture = convs
        self.fc_architecture = fcs
        self.use_lstm = use_lstm
        self.padding_type = padding
        self.is_continuous = continuous
        self.cpg_context_size = context_size
        self.cpg_network_shape = cpg_network_shape
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_train_stats = batch_norm_train_stats
        self.dropout = dropout
        if context_size is not None:
            self.context_vector = tf.get_variable(name='context_vector',
                                                  dtype=tf.float32,
                                                  shape=[1, context_size],
                                                  initializer=tf.initializers.random_uniform())
        else:
            self.context_vector = None

    def run_network(self,
                    inpt,
                    masks,
                    rnn_state,
                    num_actions,
                    lstm_unit,
                    nenvs,
                    step_size,
                    scope,
                    is_train):

        if self.is_continuous:
            return self.mlp_network()
        else:
            return self.cnn_network(inpt, masks, rnn_state, num_actions,
                                    lstm_unit, nenvs, step_size, scope, is_train=is_train)

    def mlp_network(self):
        raise NotImplementedError('Continuous episodes are not supported yet!')

    def cnn_network(self,
                    inpt,
                    masks,
                    rnn_state,
                    num_actions,
                    lstm_unit,
                    nenvs,
                    step_size,
                    scope,
                    is_train=False):

        with tf.variable_scope('cnn_conv_base_network', reuse=tf.AUTO_REUSE):
            conv_name = 'conv_base_params'
            if conv_name not in self.all_variables:
                cnn_base_params = create_cnn_network_params(context_dim=self.cpg_context_size,
                                                            cnn_architecture=self.conv_architecture,
                                                            padding=self.padding_type,
                                                            initializer=tf.orthogonal_initializer(np.sqrt(2.0)),
                                                            name=conv_name,
                                                            cpg_network_shape=self.cpg_network_shape,
                                                            dropout=self.dropout,
                                                            use_batch_norm=self.use_batch_norm,
                                                            batch_norm_momentum=self.batch_norm_momentum,
                                                            batch_norm_train_stats=self.batch_norm_train_stats)
                self.all_variables.update(cnn_base_params)

            conv_output = cnn_network(inpt,
                                      self.all_variables[conv_name],
                                      gen_vector=self.context_vector,
                                      is_train=is_train)
            conv_flattened = layers.flatten(conv_output)

        with tf.variable_scope('cnn_fc_base_network', reuse=tf.AUTO_REUSE):
            fc_name = 'cnn_fc'
            if fc_name not in self.all_variables:
                fc_base_params = create_fc_network_params(input_dim=tf.shape(conv_flattened)[-1],
                                                          context_dim=self.cpg_context_size,
                                                          fc_architecture=self.fc_architecture,
                                                          initializer=tf.orthogonal_initializer(np.sqrt(2.0)),
                                                          name=fc_name,
                                                          cpg_network_shape=self.cpg_network_shape,
                                                          dropout=self.dropout,
                                                          use_batch_norm=self.use_batch_norm,
                                                          batch_norm_momentum=self.batch_norm_momentum,
                                                          batch_norm_train_stats=self.batch_norm_train_stats)
                self.all_variables.update(fc_base_params)

            fc_output = fc_network(input=conv_flattened,
                                   fc_params=self.all_variables[fc_name],
                                   gen_vector=self.context_vector,
                                   is_train=is_train)

        with tf.variable_scope('cnn_rnn_base_network', reuse=tf.AUTO_REUSE):
            rnn_name = 'rnn'
            if rnn_name not in self.all_variables:
                rnn_base_params = create_lstm_network_params(input_dim=tf.shape(fc_output)[-1],
                                                             context_dim=self.cpg_context_size,
                                                             hidden_dim=lstm_unit,
                                                             initializer=tf.orthogonal_initializer(np.sqrt(2.0)),
                                                             name=rnn_name,
                                                             cpg_network_shape=self.cpg_network_shape,
                                                             dropout=self.dropout,
                                                             use_batch_norm=self.use_batch_norm,
                                                             batch_norm_momentum=self.batch_norm_momentum,
                                                             batch_norm_train_stats=self.batch_norm_train_stats)
                self.all_variables.update(rnn_base_params)

            rnn_in = batch_to_seq(fc_output, nenvs, step_size)
            masks = batch_to_seq(masks, nenvs, step_size)
            rnn_output, rnn_state = lstm_network(inputs=rnn_in,
                                                 keep_props=masks,
                                                 state=rnn_state,
                                                 lstm_params=self.all_variables[rnn_name],
                                                 gen_vector=self.context_vector,
                                                 is_train=is_train)
            rnn_output = seq_to_batch(rnn_output, nenvs, step_size)

        if self.use_lstm:
            output = rnn_output
        else:
            output = fc_output

        with tf.variable_scope('policy_network', reuse=tf.AUTO_REUSE):
            policy_name = 'policy'
            if policy_name not in self.all_variables:
                policy_params = create_fc_network_params(input_dim=tf.shape(output)[-1],
                                                         context_dim=self.cpg_context_size,
                                                         fc_architecture=[num_actions],
                                                         initializer=tf.orthogonal_initializer(0.1),
                                                         name=policy_name,
                                                         cpg_network_shape=self.cpg_network_shape,
                                                         dropout=self.dropout,
                                                         use_batch_norm=self.use_batch_norm,
                                                         batch_norm_momentum=self.batch_norm_momentum,
                                                         batch_norm_train_stats=self.batch_norm_train_stats)
                self.all_variables.update(policy_params)

            policy = fc_network(input=output,
                                fc_params=self.all_variables[policy_name],
                                gen_vector=self.context_vector,
                                is_train=is_train,
                                activation=None)

            dist = tf.distributions.Categorical(probs=tf.nn.softmax(policy))

        with tf.variable_scope('value_network', reuse=tf.AUTO_REUSE):
            value_name = 'value'
            if value_name not in self.all_variables:
                value_params = create_fc_network_params(input_dim=tf.shape(output)[-1],
                                                        context_dim=self.cpg_context_size,
                                                        fc_architecture=[1],
                                                        initializer=tf.orthogonal_initializer(1.0),
                                                        name=value_name,
                                                        cpg_network_shape=self.cpg_network_shape,
                                                        dropout=self.dropout,
                                                        use_batch_norm=self.use_batch_norm,
                                                        batch_norm_momentum=self.batch_norm_momentum,
                                                        batch_norm_train_stats=self.batch_norm_train_stats)
                self.all_variables.update(value_params)

            value = fc_network(input=output,
                               fc_params=[1],
                               gen_vector=self.context_vector,
                               is_train=is_train,
                               activation=None)

        return dist, value, rnn_state


