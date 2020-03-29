# The purpose of this file is to prototype code for the GloVe algorithm in tensorflow2
# As soon as it is working, we would like to refactor it to use the keras style code we are developing

import numpy as np
import collections
import xn2v
from scipy.sparse import lil_matrix
import random
import tensorflow as tf
assert tf.__version__ >= "2.0"

# download text from Gutenberg for testing
local_file = 'dickens.txt'

encoder = xn2v.text_encoder.TextEncoder(local_file)
data, count, dictionary, reverse_dictionary = encoder.build_dataset()
print("Extracted a dataset with %d words" % len(data))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

vocabulary_size = min(50000, len(count))

data_index = 0


def generate_batch(batch_size, window_size):
    # data_index is updated by 1 everytime we read a data point
    global data_index

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    weights = np.ndarray(shape=(batch_size), dtype=np.float32)

    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    span = 2 * window_size + 1

    # The buffer holds the data contained within the span
    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # This is the number of context words we sample for a single target word
    num_samples = 2 * window_size

    # We break the batch reading into two for loops
    # The inner for loop fills in the batch and labels with
    # num_samples data points using data contained withing the span
    # The outper for loop repeat this for batch_size//num_samples times
    # to produce a full batch
    for i in range(batch_size // num_samples):
        k = 0
        # avoid the target word itself as a prediction
        # fill in batch and label numpy arrays
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            weights[i * num_samples + k] = abs(1.0 / (j - window_size))
            k += 1

            # Everytime we read num_samples data points,
        # we have created the maximum number of datapoints possible
        # withing a single span, so we need to move the span by 1
        # to create a fresh new span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, weights


print('data:', [reverse_dictionary[di] for di in data[:8]])


# Creating the Word Co-Occurance Matrix
# We are creating the co-occurance matrix as a compressed sparse colum matrix from scipy.
cooc_data_index = 0
dataset_size = len(data)  # We iterate through the full text
skip_window = 4  # How many words to consider left and right.

# The sparse matrix that stores the word co-occurences
cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)

print(cooc_mat.shape)


def generate_cooc(batch_size, skip_window):
    '''
    Generate co-occurence matrix by processing batches of data
    '''
    data_index = 0
    print('Running %d iterations to compute the co-occurance matrix' % (dataset_size // batch_size))
    for i in range(dataset_size // batch_size):
        # Printing progress
        if i > 0 and i % 100000 == 0:
            print('\tFinished %d iterations' % i)

        # Generating a single batch of data
        batch, labels, weights = generate_batch(batch_size, skip_window)
        labels = labels.reshape(-1)

        # Incrementing the sparse matrix entries accordingly
        for inp, lbl, w in zip(batch, labels, weights):
            cooc_mat[inp, lbl] += (1.0 * w)


# Generate the matrix
generate_cooc(8, skip_window)

# Just printing some parts of co-occurance matrix
print('Sample chunks of co-occurance matrix')

# Basically calculates the highest cooccurance of several chosen word
for i in range(10):
    idx_target = i

    # get the ith row of the sparse matrix and make it dense
    ith_row = cooc_mat.getrow(idx_target)
    ith_row_dense = ith_row.toarray('C').reshape(-1)

    # select target words only with a reasonable words around it.
    while np.sum(ith_row_dense) < 10 or np.sum(ith_row_dense) > 50000:
        # Choose a random word
        idx_target = np.random.randint(0, vocabulary_size)

        # get the ith row of the sparse matrix and make it dense
        ith_row = cooc_mat.getrow(idx_target)
        ith_row_dense = ith_row.toarray('C').reshape(-1)

    print('\nTarget Word: "%s"' % reverse_dictionary[idx_target])

    sort_indices = np.argsort(ith_row_dense).reshape(-1)  # indices with highest count of ith_row_dense
    sort_indices = np.flip(sort_indices, axis=0)  # reverse the array (to get max values to the start)

    # printing several context words to make sure cooc_mat is correct
    print('Context word:', end='')
    for j in range(10):
        idx_context = sort_indices[j]
        print('"%s"(id:%d,count:%.2f), ' % (reverse_dictionary[idx_context], idx_context, ith_row_dense[idx_context]),
              end='')
    print()

for window_size in [2, 4]:
    data_index = 0
    batch, labels, weights = generate_batch(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' % window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    print('    weights:', [w for w in weights])


# Defining Hyperparameters
batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4 # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors
valid_size = 16 # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 50

# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

num_sampled = 32 # Number of negative examples to sample.

epsilon = 1 # used for the stability of log in the loss function

####### START HERE ADAPT TO Tensorflow2

# Defining Inputs and Outputs

# tf.reset_default_graph()

# Training input data (target word IDs).
#train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# Training input label data (context word IDs)
#train_labels = tf.placeholder(tf.int32, shape=[batch_size])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


# Variables.
in_embeddings = tf.Variable(
    tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='in.embeddings')
in_bias_embeddings = tf.Variable(tf.random.uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='in.embeddings_bias')

out_embeddings = tf.Variable(
    tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='out.embeddings')
out_bias_embeddings = tf.Variable(tf.random.uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='out.embeddings_bias')


def get_embeds(train_dataset, train_labels):
    # Look up embeddings for inputs and outputs
    # Have two seperate embedding vector spaces for inputs and outputs
    embed_in = tf.nn.embedding_lookup(in_embeddings, train_dataset)
    embed_out = tf.nn.embedding_lookup(out_embeddings, train_labels)
    embed_bias_in = tf.nn.embedding_lookup(in_bias_embeddings, train_dataset)
    embed_bias_out = tf.nn.embedding_lookup(out_bias_embeddings, train_labels)

    return embed_in, embed_out, embed_bias_in, embed_bias_out

# weights used in the cost function
# weights_x = tf.placeholder(tf.float32,shape=[batch_size],name='weights_x')
# Cooccurence value for that position
# x_ij = tf.placeholder(tf.float32,shape=[batch_size],name='x_ij')

# Compute the loss defined in the paper. Note that
# I'm not following the exact equation given (which is computing a pair of words at a time)
# I'm calculating the loss for a batch at one time, but the calculations are identical.
# I also made an assumption about the bias, that it is a smaller type of embedding

def get_loss(weights_x, x_ij, embed_in, embed_out, embed_bias_in, embed_bias_out):
    loss = tf.reduce_mean(
        weights_x * (tf.reduce_sum(embed_in*embed_out,axis=1) + embed_bias_in + embed_bias_out - tf.math.log(epsilon+x_ij))**2)
    return loss

def get_similarity():
    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    embeddings = (in_embeddings + out_embeddings) / 2.0
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    return similarity

# Optimizer.
optimizer = tf.optimizers.Adagrad(1.0)


 # Optimization process.

#_, l = session.run([optimizer, loss], feed_dict=feed_dict)
def run_optimization(train_dataset, train_labels, weights_x, x_ij):
    with tf.device('/cpu:0'):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            embed_in, embed_out, embed_bias_in, embed_bias_out = get_embeds(train_dataset, train_labels)
            loss = get_loss(weights_x, x_ij, embed_in, embed_out, embed_bias_in, embed_bias_out)

        gradients = g.gradient(loss, [embed_in, embed_out, embed_bias_in, embed_bias_out])
        optimizer.apply_gradients(zip(gradients, [embed_in, embed_out, embed_bias_in, embed_bias_out]))
        return loss

num_steps = 100001
glove_loss = []

average_loss = 0
#with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
 #   tf.global_variables_initializer().run()
print('Starting optimization')

for step in range(num_steps):
    batch_data, batch_labels, batch_weights = generate_batch(batch_size, skip_window)

    # Computing the weights required by the loss function
    batch_weights = []  # weighting used in the loss function
    batch_xij = []  # weighted frequency of finding i near j

    # Compute the weights for each datapoint in the batch
    for inp, lbl in zip(batch_data, batch_labels.reshape(-1)):
        point_weight = (cooc_mat[inp, lbl] / 100.0) ** 0.75 if cooc_mat[inp, lbl] < 100.0 else 1.0
        batch_weights.append(point_weight)
        batch_xij.append(cooc_mat[inp, lbl])
    batch_weights = np.clip(batch_weights, -100, 1)
    batch_xij = np.asarray(batch_xij)

    # Populate the feed_dict and run the optimizer (minimize loss)
    # and compute the loss. Specifically we provide
    # train_dataset/train_labels: training inputs and training labels
    # weights_x: measures the importance of a data point with respect to how much those two words co-occur
    # x_ij: co-occurence matrix value for the row and column denoted by the words in a datapoint

    ls = run_optimization(batch_data.reshape(-1),batch_labels.reshape(-1),batch_weights,batch_xij)



    # Update the average loss variable
    average_loss += ls
    if step % 2000 == 0:
        if step > 0:
            average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step %d: %f' % (step, average_loss))
        glove_loss.append(average_loss)
        average_loss = 0

    # Here we compute the top_k closest words for a given validation word
    # in terms of the cosine distance
    # We do this for all the words in the validation set
    # Note: This is an expensive step
    if step % 10000 == 0:
        sim = get_similarity()
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
            print(log)


embeddings = (in_embeddings + out_embeddings) / 2.0
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
final_embeddings = normalized_embeddings.eval()





