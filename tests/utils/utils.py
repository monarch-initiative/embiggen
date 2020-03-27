import numpy as np


def calculate_total_probs(j, q):
    """
    Use the alias method to calculate the total probabilites of the discrete events
    :param j: alias vector
    :param q: alias-method probabilities
    :return:
    """
    N = len(j)
    probs = np.zeros(N)
    for i in range(N):
        p = q[i]
        probs[i] += p
        if p < 1.0:
            alias_index = j[i]
            probs[alias_index] += 1 - p
    s = np.sum(probs)
    return probs / s
