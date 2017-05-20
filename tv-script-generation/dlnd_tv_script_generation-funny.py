# coding: utf-8

# # Check Point
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests
from tensorflow.contrib.tensorboard.plugins import projector
import os
LOG_DIR = '.'

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# ## Build the Neural Network
# You'll build the components necessary to build a RNN by implementing the following functions below:
# - get_inputs
# - get_init_cell
# - get_embed
# - build_rnn
# - build_nn
# - get_batches
# 
# ### Check the Version of TensorFlow and Access to GPU

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))


# Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.
# It should create the following placeholders:
# - Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
# - Targets placeholder
# - Learning Rate placeholder
# 
# Return the placeholders in the following the tuple `(Input, Targets, LearingRate)`

# In[ ]:

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, shape=[None, None], name="input")
    targets = tf.placeholder(tf.int32, shape=[None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return input, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)


# ### Build RNN Cell and Initialize
# Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
# - The Rnn size should be set using `rnn_size`
# - Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
#     - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the cell and initial state in the following tuple `(Cell, InitialState)`

# In[ ]:

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    num_layers = 2
    keep_prob = 0.5
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)

    initial_state = rnn_cell.zero_state(batch_size, tf.float32)
    return rnn_cell, tf.identity(initial_state, name="initial_state")


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)


# ### Word Embedding
# Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.

# In[ ]:

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    with tf.variable_scope("embedding"):
        # embedding = tf.get_variable("embedding", [vocab_size, embed_dim], dtype=tf.float32)
        embedding_var = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1, name="word_embedding"))

        # input: batch_size * time_step * embedding_feature
        embedded_input = tf.nn.embedding_lookup(embedding_var, input_data, name="embedded_input")
    return embedded_input


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


# ### Build RNN
# You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
# - Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
#  - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)
# 
# Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 

# In[ ]:

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=initial_state)
    # rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(inputs, inputs.shape[2], 1)]

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return (outputs, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


# ### Build the Neural Network
# Apply the functions you implemented above to:
# - Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
# - Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
# - Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.
# 
# Return the logits and final state in the following tuple (Logits, FinalState) 

# In[ ]:

def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embed_dim = 400

    inputs = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, inputs)

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

    return (logits, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)


# ### Batches
# Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
# - The first element is a single batch of **input** with the shape `[batch size, sequence length]`
# - The second element is a single batch of **targets** with the shape `[batch size, sequence length]`
# 
# If you can't fill the last batch with enough data, drop the last batch.
# 
# For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
# ```
# [
#   # First Batch
#   [
#     # Batch of Input
#     [[ 1  2  3], [ 7  8  9]],
#     # Batch of targets
#     [[ 2  3  4], [ 8  9 10]]
#   ],
#  
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 4  5  6], [10 11 12]],
#     # Batch of targets
#     [[ 5  6  7], [11 12 13]]
#   ]
# ]
# ```

# In[ ]:

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    '''
        1. make sure the last word will not be None in each batch. So make len(int_text) - 1
        2. each batch windows = batch_size * seq_length
    '''
    n_batches = int((len(int_text) - 1) / (batch_size * seq_length))

    '''
        prepare the x_data, y_data
    '''
    x_data = np.array(int_text[: n_batches * batch_size * seq_length])
    y_data = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    # convert the array to be [batch_size, None]
    temp_x = x_data.reshape(batch_size, -1)
    temp_y = y_data.reshape(batch_size, -1)

    # convert the array to be n_batches sub-list of [sub-array[batch_size, None]]
    x_batches = np.split(temp_x, n_batches, 1)
    y_batches = np.split(temp_y, n_batches, 1)

    # x_batches = np.split(x_data.reshape(batch_size, -1), n_batches, 1)
    # y_batches = np.split(y_data.reshape(batch_size, -1), n_batches, 1)

    batches_np = np.array(list(zip(x_batches, y_batches)))

    return batches_np


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)

# ## Neural Network Training
# ### Hyperparameters
# Tune the following parameters:
# 
# - Set `num_epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `seq_length` to the length of sequence.
# - Set `learning_rate` to the learning rate.
# - Set `show_every_n_batches` to the number of batches the neural network should print progress.

# In[ ]:

# Number of Epochs
num_epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 10

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'

# ### Build the Graph
# Build the graph using the neural network you implemented.

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)

# ## Train
# Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    # Save Model
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # Add embedding tensorboard visualization
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/word_embedding:0"
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    projector.visualize_embeddings(summary_writer, config)


    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    saver.save(sess, save_dir)

    print('Model Trained and Saved')

# ## Save Parameters
# Save `seq_length` and `save_dir` for generating a new TV script.

# In[ ]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))

# # Checkpoint

# In[ ]:
