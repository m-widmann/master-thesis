import pickle
from math import sqrt
import numpy as np
from scipy.interpolate import interp1d

def best_rectangle(n, max_cols=None):
    #usefull for plotting subplots
    n = int(n)
    if n == 0:
        return (0, 0)
    if max_cols is None or max_cols < 1:
        max_cols = int(round(sqrt(n),0))

    best_rows, best_cols = n, 1  # Default to tallest rectangle

    for cols in range(max_cols, 0, -1):  # Iterate from max_cols down to 1
        rows = (n + cols - 1) // cols  # Ceiling division
        if rows * cols >= n:
            # Update only if this is the flattest so far
            if rows <= best_rows:
                best_rows, best_cols = rows, cols
                break  # Flattest possible for this n and max_cols
    return best_rows, best_cols

def pickle_save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def pickle_load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def split_array_along_axis(arr, n, axis):
    #split arr into n pieces along the axis, interpolate if split is not even
    arr = np.asarray(arr)
    length = arr.shape[axis]
    target_length = int(np.ceil(length / n) * n)

    if target_length != length:
        x_old = np.linspace(0, 1, length)
        x_new = np.linspace(0, 1, target_length)
        f = interp1d(x_old, arr, axis=axis, kind='linear')
        arr = f(x_new)
    split = np.split(arr, n, axis=axis)
    return np.stack(split, axis=axis)