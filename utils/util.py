import numpy as np


HBAR = 6.58211928*(10**(-19)) # keV * seconds
SPEED_OF_LIGHT = 2.99792*(10**18) # Angstroms / second


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

def angle_convert(q_norm, energy):
    """Convert momentum transfer values to angle. 

    Note:  This returns 2-theta.

    Args:  
        q_norm:  (array) An array of MT values
        energy:  (float) The photon energy in keV
    
    Returns:  
        An array of angle values.
    """
    q_norm = np.array([q for q in q_norm if np.abs((HBAR * SPEED_OF_LIGHT * q) / (2 * energy)) <= 1])
    return 2 * (180.0 / np.pi) * np.arcsin((HBAR * SPEED_OF_LIGHT * q_norm) / (2 * energy))
