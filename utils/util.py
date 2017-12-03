import numpy as np


def norm(x, mu, std=1.0):
    """Calculates an unnormalized gaussian distribution over the input array.
    
    Args:  
        x:  (array) the independent variable of the distribution
        mu:  (float) the center of the distribution
        std:  (float, optional) the standard deviation
    Returns:  
        An array of norm values over x
    """
    return np.exp(-(x - mu)**2 / (2 * std**2))