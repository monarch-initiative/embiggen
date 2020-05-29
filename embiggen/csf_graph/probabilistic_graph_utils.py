import numpy as np
from numba import njit
from typing import Tuple

@njit
def alias_setup(probabilities : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    The alias method allows sampling from a discrete distribution in O(1) time.
    This is used here to perform biased random walks more efficiently.

    Parameters
    ----------
    probabilities:np.ndarray
        The normalized probabilities, e.g., [0.4 0.28 0.32], for
        transition to each neighbor.

    Returns
    -------
    parameters:Tuple[np.ndarray, np.ndarray]
        Tuple of the parameters needed for the extraction. 
        The first argument is the mapping to the less probable binary outcome,
        and the second is the uniform distribution over binary outcomes
    """
    
    # find the values bigger and smaller than 1/k
    # this is done this way to have better numerical accuracy
    q = probabilities * probabilities.size
    smaller_mask = q < 1.0
    smaller = list(np.where(smaller_mask)[0])
    larger  = list(np.where(~smaller_mask)[0])

    # j is the mapping of the opposite event in the Bernulli trias
    j = np.zeros_like(probabilities)
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
def alias_draw(parameters : Tuple[np.ndarray, np.ndarray]) -> int:
    """Draw sample from a non-uniform discrete distribution using alias sampling.

    Parameters
    ----------
    parameters:Tuple[np.ndarray, np.ndarray]
        Tuple of the parameters needed for the extraction. 
        The first argument is the mapping to the less probable binary outcome,
        and the second is the uniform distribution over binary outcomes

    Returns:
    index:int 
        index of random sample from non-uniform discrete distribution

    """
    j, q = parameters
    # extract a random index for the mixture
    index = np.random.randint(0, len())
    # do the Bernulli trial
    if np.random.rand() < q[index]:
        # most probable case and fastest
        return index
    return j[index]