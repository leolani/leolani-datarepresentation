import numpy as np

from constants import CAPITALIZED_TYPES


def hash_claim_id(triple):
    return '_'.join(triple)


def is_proper_noun(types):
    return any(i in types for i in CAPITALIZED_TYPES)


def casefold_text(text, format='triple'):
    if format == 'triple':
        return text.strip().lower().replace(" ", "-") if isinstance(text, basestring) else text
    elif format == 'natural':
        return text.strip().lower().replace("-", " ") if isinstance(text, basestring) else text
    else:
        return text


def spherical2cartesian(phi, theta, depth):
    """
    Spherical Coordinates to Cartesian Coordinates

    Phi: Left to Right, Theta: Down to Up, Depth: Distance
    x: Left to Right, y: down to up, z: close to far

    Parameters
    ----------
    phi: float
    theta: float
    depth: float

    Returns
    -------
    x,y,z: float, float, float

    """
    x = depth * np.sin(theta) * np.cos(phi)
    y = depth * np.cos(theta)
    z = depth * np.sin(theta) * np.sin(phi)

    return x, y, z
