import numpy as np
import h5py

import pybind11_analysis


# Change: use custom C++ extension via pybind11 (see /pybind11_extension)


def load_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        return f["probabilities"][:]


def colorcode_probabilities(probabilities):
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    colored_probabilities = pybind11_analysis.colorcode_probabilities(probabilities, class_colors)
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    probabilities = load_probabilities(folder)
    return colorcode_probabilities(probabilities)
