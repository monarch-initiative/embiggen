# The purpose of this file is to prototype code for the GloVe algorithm in tensorflow2
# As soon as it is working, we would like to refactor it to use the keras style code we are developing

import numpy as np
import xn2v

local_file = 'tests/twocities.txt'

encoder = xn2v.text_encoder.TextEncoder(local_file)
data, count, dictionary, reverse_dictionary = encoder.build_dataset()
print("Extracted a dataset with %d words" % len(data))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

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

for window_size in [2, 4]:
    data_index = 0
    batch, labels, weights = generate_batch(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' % window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    print('    weights:', [w for w in weights])