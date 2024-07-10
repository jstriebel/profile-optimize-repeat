import numpy as np
import h5py

import pybind11_analysis


# Change: combine custom C++ extension with chunking


CHUNK_SIZE = 1024


def get_chunk_slices(length, chunk_size):
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        yield slice(start, end)


def load_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        x_len, y_len = f["probabilities"].shape[:2]
        for x_slice in get_chunk_slices(x_len, CHUNK_SIZE):
            for y_slice in get_chunk_slices(y_len, CHUNK_SIZE):
                yield (x_slice, y_slice), f["probabilities"][x_slice, y_slice]


def colorcode_probabilities(probabilities):
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    colored_probabilities = pybind11_analysis.colorcode_probabilities(probabilities, class_colors)
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        x_len, y_len = f["probabilities"].shape[:2]
    colored_probabilities = np.zeros((x_len, y_len, 3), dtype=np.uint8)
    for chunk_slice, chunk in load_probabilities(folder):
        colored_probabilities[chunk_slice] = colorcode_probabilities(chunk)
    return colored_probabilities
