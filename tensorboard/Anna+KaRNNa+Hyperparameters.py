
# coding: utf-8

# # Anna KaRNNa
# 
# In this notebook, I'll build a character-wise RNN trained on Anna Karenina, one of my all-time favorite books. It'll be able to generate new text based on the text from the book.
# 
# This network is based off of Andrej Karpathy's [post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [implementation in Torch](https://github.com/karpathy/char-rnn). Also, some information [here at r2rt](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html) and from [Sherjil Ozair](https://github.com/sherjilozair/char-rnn-tensorflow) on GitHub. Below is the general architecture of the character-wise RNN.
# 
# <img src="assets/charseq.jpeg" width="500">

# In[1]:

import time
from collections import namedtuple

import numpy as np
import tensorflow as tf


# First we'll load the text file and convert it into integers for our network to use.

# In[2]:

with open('anna.txt', 'r') as f:
    text=f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


# In[3]:

text[:100]


# In[4]:

chars[:100]


# Now I need to split up the data into batches, and into training and validation sets. I should be making a test set here, but I'm not going to worry about that. My test will be if the network can generate new text.
# 
# Here I'll make both input and target arrays. The targets are the same as the inputs, except shifted one character over. I'll also drop the last bit of data so that I'll only have completely full batches.
# 
# The idea here is to make a 2D matrix where the number of rows is equal to the number of batches. Each row will be one long concatenated string from the character data. We'll split this data into a training set and validation set using the `split_frac` keyword. This will keep 90% of the batches in the training set, the other 10% in the validation set.

# In[5]:

def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """ 
    Split character data into training and validation sets, inputs and targets for each set.
    
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    
    
    Returns train_x, train_y, val_x, val_y
    """
    
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the virst split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    
    return train_x, train_y, val_x, val_y


# In[6]:

train_x, train_y, val_x, val_y = split_data(chars, 10, 200)


# In[7]:

train_x.shape


# In[8]:

train_x[:,:10]


# I'll write another function to grab batches out of the arrays made by split data. Here each batch will be a sliding window on these arrays with size `batch_size X num_steps`. For example, if we want our network to train on a sequence of 100 characters, `num_steps = 100`. For the next batch, we'll shift this window the next sequence of `num_steps` characters. In this way we can feed batches to the network and the cell states will continue through on each batch.

# In[9]:

def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]


# In[10]:

def build_rnn(num_classes, batch_size=50, num_steps=50, lstm_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
        
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()
    
    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        x_one_hot = tf.one_hot(inputs, num_classes, name='x_one_hot')
    
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
        y_one_hot = tf.one_hot(targets, num_classes, name='y_one_hot')
        y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # Build the RNN layers
    with tf.name_scope("RNN_cells"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
        outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=initial_state)
    
    final_state = state
    
    # Reshape output so it's a bunch of rows, one row for each cell output
    with tf.name_scope('sequence_reshape'):
        seq_output = tf.concat(outputs, axis=1,name='seq_output')
        output = tf.reshape(seq_output, [-1, lstm_size], name='graph_output')
    
    # Now connect the RNN outputs to a softmax layer and calculate the cost
    with tf.name_scope('logits'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1),
                               name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(num_classes), name='softmax_b')
        logits = tf.matmul(output, softmax_w) + softmax_b
        tf.summary.histogram('softmax_w', softmax_w)
        tf.summary.histogram('softmax_b', softmax_b)

    with tf.name_scope('predictions'):
        preds = tf.nn.softmax(logits, name='predictions')
        tf.summary.histogram('predictions', preds)
    
    with tf.name_scope('cost'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped, name='loss')
        cost = tf.reduce_mean(loss, name='cost')
        tf.summary.scalar('cost', cost)

    # Optimizer for training, using gradient clipping to control exploding gradients
    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    merged = tf.summary.merge_all()
    
    # Export the nodes 
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph


# ## Hyperparameters
# 
# Here I'm defining the hyperparameters for the network. The two you probably haven't seen before are `lstm_size` and `num_layers`. These set the number of hidden units in the LSTM layers and the number of LSTM layers, respectively. Of course, making these bigger will improve the network's performance but you'll have to watch out for overfitting. If your validation loss is much larger than the training loss, you're probably overfitting. Decrease the size of the network or decrease the dropout keep probability.

# In[11]:

batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001


# ## Training
# 
# Time for training which is is pretty straightforward. Here I pass in some data, and get an LSTM state back. Then I pass that state back in to the network so the next batch can continue the state from the previous batch. And every so often (set by `save_every_n`) I calculate the validation loss and save a checkpoint.

# In[12]:

# get_ipython().system('mkdir -p checkpoints/anna')


# In[13]:

def train(model, epochs, file_writer):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Use the line below to load a checkpoint and resume training
        #saver.restore(sess, 'checkpoints/anna20.ckpt')

        n_batches = int(train_x.shape[1]/num_steps)
        iterations = n_batches * epochs
        for e in range(epochs):

            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
                iteration = e*n_batches + b
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: 0.5,
                        model.initial_state: new_state}
                summary, batch_loss, new_state, _ = sess.run([model.merged, model.cost, 
                                                              model.final_state, model.optimizer], 
                                                              feed_dict=feed)
                loss += batch_loss
                end = time.time()
                print('Epoch {}/{} '.format(e+1, epochs),
                      'Iteration {}/{}'.format(iteration, iterations),
                      'Training loss: {:.4f}'.format(loss/b),
                      '{:.4f} sec/batch'.format((end-start)))

                file_writer.add_summary(summary, iteration)


# In[ ]:

epochs = 20
batch_size = 100
num_steps = 100
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

for lstm_size in [128,256,512]:
    for num_layers in [1, 2]:
        for learning_rate in [0.002, 0.001]:
            log_string = 'logs/4/lr={},rl={},ru={}'.format(learning_rate, num_layers, lstm_size)
            writer = tf.summary.FileWriter(log_string)
            model = build_rnn(len(vocab), 
                    batch_size=batch_size,
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    lstm_size=lstm_size,
                    num_layers=num_layers)
            
            train(model, epochs, writer)


# In[ ]:

tf.train.get_checkpoint_state('checkpoints/anna')


# ## Sampling
# 
# Now that the network is trained, we'll can use it to generate new text. The idea is that we pass in a character, then the network will predict the next character. We can use the new one, to predict the next one. And we keep doing this to generate all new text. I also included some functionality to prime the network with some text by passing in a string and building up a state from that.
# 
# The network gives us predictions for each character. To reduce noise and make things a little less random, I'm going to only choose a new character from the top N most likely characters.
# 
# 

# In[17]:

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


# In[41]:

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    prime = "Far"
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)


# In[44]:

checkpoint = "checkpoints/anna/i3560_l512_1.122.ckpt"
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[43]:

checkpoint = "checkpoints/anna/i200_l512_2.432.ckpt"
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[46]:

checkpoint = "checkpoints/anna/i600_l512_1.750.ckpt"
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)


# In[47]:

checkpoint = "checkpoints/anna/i1000_l512_1.484.ckpt"
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

