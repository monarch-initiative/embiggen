import numpy as np  # type:ignore


def calculate_total_probs(j: np.ndarray, q: np.ndarray) -> np.array:
    """Use the alias method to calculate the total probabilities of the discrete events.

    Args:
        j: A vector of node aliases (e.g. array([2, 3, 0, 2])).
        q: A vector of alias-method probabilities (e.g. array([0.5, 0.5, 1. , 0.5])).

    Returns:
        An array of floats representing the total probabilities of the discrete event (e.g. array([0.125, 0.125,
        0.5  , 0.25 ])).
    """

    n = len(j)
    probs = np.zeros(n)

    for i in range(n):
        p = q[i]
        probs[i] += p

        if p < 1.0:
            alias_index = j[i]
            probs[alias_index] += 1 - p

    s = np.sum(probs)

    return probs / s
