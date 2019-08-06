from utils import *


# Unit Tests

def test_create_fc_network_params():
    name = 'fc_network'
    params = create_fc_network_params(input_dim=10,
                                      context_dim=None,
                                      fc_architecture=[15, 1],
                                      initializer=None,
                                      name=name,
                                      cpg_network_shape=None,
                                      dropout=.5,
                                      use_batch_norm=True,
                                      batch_norm_momentum=.1,
                                      batch_norm_train_stats=True)
    name2 = name + '_2'
    params2 = create_fc_network_params(input_dim=10,
                                      context_dim=10,
                                      fc_architecture=[15, 1],
                                      initializer=None,
                                      name=name2,
                                      cpg_network_shape=[10],
                                      dropout=.5,
                                      use_batch_norm=True,
                                      batch_norm_momentum=.1,
                                      batch_norm_train_stats=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Testing FC network param creation....')
    assert params[name]['weights'] != [], 'Weights is empty list despite non-empty architecture'
    assert params[name]['bias'] != [], 'Bias is empty list despite non-empty architecture'
    assert params2[name2]['weights'] != [], 'Weights is empty list despite non-empty architecture'
    assert params2[name2]['bias'] != [], 'Bias is empty list despite non-empty architecture'
    print('Valid!')


def test_create_cnn_network_params():
    name = 'cnn_network'
    params = create_cnn_network_params(context_dim=None,
                                       cnn_architecture=[(2, 3, 2), (1, 1, 1)],
                                       padding='VALID',
                                       initializer=None,
                                       name=name,
                                       cpg_network_shape=None,
                                       dropout=.5,
                                       use_batch_norm=True,
                                       batch_norm_momentum=.1,
                                       batch_norm_train_stats=True)
    name2 = 'cnn_network_2'
    params2 = create_cnn_network_params(context_dim=10,
                                       cnn_architecture=[(2, 3, 2), (1, 1, 1)],
                                       padding='VALID',
                                       initializer=None,
                                       name=name2,
                                       cpg_network_shape=[],
                                       dropout=.5,
                                       use_batch_norm=True,
                                       batch_norm_momentum=.1,
                                       batch_norm_train_stats=True)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Testing CNN network param creation...')
    assert params['cnn_network']['filters'] != [], 'Filters empty list despite non-empty architecture'
    assert params['cnn_network']['bias'] != [], 'Bias empty list despite non-empty architecture'
    assert params['cnn_network']['strides'] != [], 'Strides empty list despite non-empty architecture'
    assert params['cnn_network']['padding'] == 'VALID', 'Padding should be VALID'
    print('Valid!')


def test_create_lstm_network_params():
    name = 'lstm_network'
    params = create_lstm_network_params(input_dim=10,
                                        hidden_dim=10,
                                        context_dim=None,
                                        initializer=None,
                                        name=name,
                                        cpg_network_shape=None,
                                        dropout=.5,
                                        use_batch_norm=True,
                                        batch_norm_momentum=.1,
                                        batch_norm_train_stats=True,
                                        init_scale=1.0)

    name2 = name + '_2'
    params2 = create_lstm_network_params(input_dim=10,
                                        hidden_dim=10,
                                        context_dim=10,
                                        initializer=None,
                                        name=name,
                                        cpg_network_shape=[],
                                        dropout=.5,
                                        use_batch_norm=True,
                                        batch_norm_momentum=.1,
                                        batch_norm_train_stats=True,
                                        init_scale=1.0)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('Testing LSTM network params creation...')
    assert params[name]['input_weights'] is not None, 'input weights cannot be none!'
    assert params[name]['hidden_weights'] is not None, 'hidden weights cannot be none!'
    assert params[name]['bias'] is not None, 'bias cannot be none!'

    assert params[name]['input_weights'] is not None, 'input weights cannot be none!'
    assert params[name]['hidden_weights'] is not None, 'hidden weights cannot be none!'
    assert params[name]['bias'] is not None, 'bias cannot be none!'
    print('Valid!')


def test_fc_network(use_pg=True):
    name = 'fc_network_params'
    if use_pg:
        gen_vector = tf.get_variable(name='gen_vector',
                                     dtype=tf.float32,
                                     shape=[1, 10],
                                     initializer=tf.random_uniform_initializer())
        cpg_network_shape = []
    else:
        gen_vector = None
        cpg_network_shape = None
    with tf.variable_scope('fc_network_testing', reuse=tf.AUTO_REUSE):
        fc_params = create_fc_network_params(input_dim=15,
                                             context_dim=10,
                                             fc_architecture=[15, 1],
                                             initializer=None,
                                             name=name,
                                             cpg_network_shape=cpg_network_shape,
                                             dropout=.5,
                                             use_batch_norm=True,
                                             batch_norm_momentum=.1,
                                             batch_norm_train_stats=True)
        input_tensor = tf.get_variable(name='input_tensor',
                                       dtype=tf.float32,
                                       shape=[2, 15],
                                       initializer=tf.random_uniform_initializer())
        output = fc_network(input=input_tensor,
                            gen_vector=gen_vector,
                            fc_params=fc_params[name],
                            is_train=False)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(output)
    print(f"Testing fc_network {'with cpg' if use_pg else 'without cpg'}...")
    assert output.shape == (2, 1), 'output has unexpected shape!'
    print('Valid!')


def test_cnn_network(use_pg=True):

    with tf.variable_scope('cnn_network_testing', reuse=tf.AUTO_REUSE):
        name = 'cnn_params'
        input_tensor = tf.get_variable(name='input_tensor',
                                       dtype=tf.float32,
                                       shape=[5, 20, 20, 3],
                                       initializer=tf.random_uniform_initializer())
        if use_pg:
            gen_vector = tf.get_variable(name='gen_vector',
                                     dtype=tf.float32,
                                     shape=[1, 4],
                                     initializer=tf.random_uniform_initializer())
            cpg_network_shape = []
        else:
            gen_vector = None
            cpg_network_shape = None

        cnn_params = create_cnn_network_params(context_dim=4,
                                               cnn_architecture=[(2, 3, 2), (1, 1, 1)],
                                               padding='VALID',
                                               initializer=None,
                                               name=name,
                                               cpg_network_shape=cpg_network_shape,
                                               dropout=.5,
                                               use_batch_norm=True,
                                               batch_norm_momentum=.1,
                                               batch_norm_train_stats=True)

        output = cnn_network(input=input_tensor,
                             cnn_params=cnn_params[name],
                             gen_vector=gen_vector,
                             is_train=False)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(output)
    print(f"Testing cnn network {'with cpg' if use_pg else 'without cpg'}...")
    assert output.shape == (5, 9, 9, 1), 'Shape is not as expected!'
    print('Valid!')


def test_lstm_network(use_pg=True):
    with tf.variable_scope('lstm_network', reuse=tf.AUTO_REUSE):
        name = 'lstm_params'
        context_dim = 4
        input_tensors = []

        hidden_tensor = tf.get_variable(name='hidden_tensor',
                                        dtype=tf.float32,
                                        shape=[5, 20],
                                        initializer=tf.random_uniform_initializer())

        for i in range(3):
            input_tensor = tf.get_variable(name='input_tensor',
                                           dtype=tf.float32,
                                           shape=[5, 10],
                                           initializer=tf.random_uniform_initializer())
            input_tensors.append(input_tensor)

        if use_pg:
            gen_vector = tf.get_variable(name='gen_vector',
                                         dtype=tf.float32,
                                         shape=[1, context_dim],
                                         initializer=tf.random_uniform_initializer())
            cpg_network_shape = []
        else:
            gen_vector = None
            cpg_network_shape = None

        lstm_params = create_lstm_network_params(input_dim=10,
                                                 hidden_dim=10,
                                                 context_dim=context_dim,
                                                 initializer=None,
                                                 name=name,
                                                 cpg_network_shape=cpg_network_shape,
                                                 dropout=.5,
                                                 use_batch_norm=True,
                                                 batch_norm_momentum=.1,
                                                 batch_norm_train_stats=True,
                                                 init_scale=1.0)

        output, hidden_state = lstm_network(inputs=input_tensors,
                                            keep_props=[1., 1., 1.],
                                            state=hidden_tensor,
                                            lstm_params=lstm_params[name],
                                            gen_vector=gen_vector,
                                            is_train=False)

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        output = sess.run(output)
        hidden_state = sess.run(hidden_state)
        print(f"Testing lstm network {'with cpg' if use_pg else 'without cpg'}...")
        # assert output.shape == (5, 9, 9, 1), 'Shape is not as expected!'
        print(list(map(lambda output_layer: output_layer.shape, output)))
        print(hidden_state.shape)
        print('Valid!')

if __name__ == '__main__':
    test_create_fc_network_params()
    test_create_cnn_network_params()
    test_create_lstm_network_params()
    test_fc_network(use_pg=True)
    test_fc_network(use_pg=False)
    test_cnn_network(use_pg=True)
    test_cnn_network(use_pg=False)
    test_lstm_network(use_pg=True)
    test_lstm_network(use_pg=False)





