import numpy as np

def norm8bit(v, min_val=None, max_val=None):
    """
    Normalize the input array to 8-bit range [0, 255].

    Args:
        v (numpy.ndarray): Input array to be normalized.
        min_val (float or None): Minimum value of the range. If None, the minimum value of the array is used.
        max_val (float or None): Maximum value of the range. If None, the maximum value of the array is used.

    Returns:
        numpy.ndarray: Normalized array in the range [0, 255] with data type uint8.

    """
    if min_val is None:
        min_val = v.min()

    if max_val is None:
        max_val = v.max()

    max_val -= min_val

    v = ((v - min_val) / max_val) * 255

    return v.astype(np.uint8)



def norm16bit(v, min_val=None, max_val=None):
    """
    Normalize the input array to 16-bit range [0, 65535].

    Args:
        v (numpy.ndarray): Input array to be normalized.
        min_val (float or None): Minimum value of the range. If None, the minimum value of the array is used.
        max_val (float or None): Maximum value of the range. If None, the maximum value of the array is used.

    Returns:
        numpy.ndarray: Normalized array in the range [0, 65535] with data type uint16.

    """
    if min_val is None:
        min_val = v.min()

    if max_val is None:
        max_val = v.max()

    max_val -= min_val

    v = ((v - min_val) / max_val) * 65535

    return v.astype(np.uint16)