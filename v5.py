import numba
import numpy as np
import h5py


# Change: numba jitting: similar to v1


def load_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        return f["probabilities"][:]


@numba.njit([numba.uint8[:, :, :](numba.float64[:, :, :])])
def colorcode_probabilities(probabilities):
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    x_len, y_len = probabilities.shape[:2]
    colored_probabilities = np.zeros((x_len, y_len, 3), dtype=np.uint8)
    for x in range(x_len):
        for y in range(y_len):
            class_index = 0
            max_prob = 0
            for i, prob in enumerate(probabilities[x, y]):
                if prob > max_prob:
                    max_prob = prob
                    class_index = i
            colored_probabilities[x, y, 0] = np.uint8(class_colors[class_index, 0] * max_prob)
            colored_probabilities[x, y, 1] = np.uint8(class_colors[class_index, 1] * max_prob)
            colored_probabilities[x, y, 2] = np.uint8(class_colors[class_index, 2] * max_prob)
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    probabilities = load_probabilities(folder)
    return colorcode_probabilities(probabilities)
