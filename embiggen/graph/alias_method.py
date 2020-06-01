import numpy as np
from numba import njit
from typing import Tuple, List


@njit
def alias_setup(probabilities: List) -> Tuple[List, List]:
    """Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    The alias method allows sampling from a discrete distribution in O(1) time.
    This is used here to perform biased random walks more efficiently.

    Parameters
    ----------
    probabilities:List
        The normalized probabilities, e.g., [0.4 0.28 0.32], for
        transition to each neighbor.

    Returns
    -------
    parameters:Tuple[List, List]
        Tuple of the parameters needed for the extraction. 
        The first argument is the mapping to the less probable binary outcome,
        and the second is the uniform distribution over binary outcomes
    """

    K = len(probabilities)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probabilities):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


@njit
def alias_draw(j: np.ndarray, q: np.ndarray) -> int:
    """Draw sample from a non-uniform discrete distribution using alias sampling.

    Parameters
    ----------
    j:np.ndarray,
        The mapping to the less probable binary outcome,
    q: np.ndarray
        Uniform distribution over binary outcomes

    Returns:
        index:int 
            index of random sample from non-uniform discrete distribution
    """
    # extract a random index for the mixture
    index = np.random.randint(0, len(q))
    # do the Bernulli trial
    if np.random.rand() < q[index]:
        # most probable case and fastest
        return index
    return j[index]
