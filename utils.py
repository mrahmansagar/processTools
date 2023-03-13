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


def ratio_above_threshold(arr, threshold, start_range=None, end_range=None):
    """
    Calculates the ratio of numbers in `my_list` that are greater than or equal to `threshold`
    and fall within the range `[start_range, end_range]`.

    Args:
        arr (list or numpy array): A list or numpy array of numbers.
        threshold (int or float): The minimum value (inclusive) for a number to be counted in the ratio.
        start_range (int or float, optional): The minimum value for a number to be considered in the ratio.
            If not provided, the minimum value in `my_list` is used.
        end_range (int or float, optional): The maximum value for a number to be considered in the ratio.
            If not provided, the maximum value in `my_list` is used.

    Returns:
        float: The ratio of numbers in `my_list` that are greater than or equal to `threshold`
        and fall within the range `[start_range, end_range]`.

    Raises:
        ValueError: If `threshold` is not provided.

    """
    if threshold is None:
        raise ValueError("threshold must be provided.")

    arr = np.ravel(arr)  # Flatten the input array if it is multidimensional.

    if start_range is None:
        start_range = np.min(arr)

    if end_range is None:
        end_range = np.max(arr)

    sub_arr = np.array([x for x in arr if start_range <= x <= end_range])

    count = np.count_nonzero(sub_arr >= threshold)

    ratio = count / len(sub_arr)

    return ratio