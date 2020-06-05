import numpy as np  # type: ignore
from numba import njit  # type: ignore
from typing import Tuple
from random import random, randint


@njit
def alias_setup(probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    The alias method allows sampling from a discrete distribution in O(1) time.
    This is used here to perform biased random walks more efficiently.

    Parameters
    ----------
    probabilities: np.ndarray
        The normalized probabilities, e.g., [0.4 0.28 0.32], for
        transition to each neighbor.

    Raises
    --------------------
    ValueError,
        If given probability vector does not sum to one.
    ValueError,
        If given probability vector is empty.

    Returns
    -------
    parameters:Tuple[List, List]
        Tuple of the parameters needed for the extraction.
        The first argument is the mapping to the less probable binary outcome,
        and the second is the uniform distribution over binary outcomes
    """
    if probabilities.size == 0:
        raise ValueError("Given probability vector is empty!")

    if abs(probabilities.sum() - 1) > 1e-8:
        raise ValueError(
            "Given probability vector does not sum to one"
        )

    q = probabilities * probabilities.size
    smaller_mask = q < 1.0
    smaller = list(np.where(smaller_mask)[0])
    larger = list(np.where(~smaller_mask)[0])

    # j is the mapping of the opposite event in the Bernulli trias
    j = np.zeros_like(probabilities, dtype=np.int16)
    # Converge to the equivalente binary mixture
    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()

        # salva il mapping del evento opposto
        j[small] = large

        # this is equibalent but has better numerical accuracy
        # q[larArgsge] = q[large] - (1.0 - q[small])
        q[large] = (q[large] + q[small]) - 1.0

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return (j, q)


@njit
def alias_draw(j: np.ndarray, q: np.ndarray) -> int:
    """Draw sample from a non-uniform discrete distribution using alias sampling.

    Parameters
    ----------
    j: np.ndarray,
        The mapping to the less probable binary outcome,
    q: np.ndarray
        Uniform distribution over binary outcomes

    Returns:
        index: int,
            index of random sample from non-uniform discrete distribution
    """
    # NB: here we are using random.random and random.randint
    # instead of the Numpy versions because in numba they are converted better.
    # extract a random index for the mixture
    index = randint(0, len(q)-1)
    # do the Bernulli trial
    if random() < q[index]:
        # most probable case and fastest
        return index
    return j[index]
