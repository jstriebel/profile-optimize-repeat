import numpy as np
import h5py


# Change: all numpy


def load_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        return f["probabilities"][:]


def colorcode_probabilities(probabilities):
    predicted_classes = np.argmax(probabilities, axis=2)
    max_prob = np.max(probabilities, axis=2)
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    colored_probabilities = class_colors[predicted_classes]
    colored_probabilities = colored_probabilities * max_prob[..., np.newaxis]
    return colored_probabilities.astype(np.uint8)


def load_and_colorcode_probabilities(folder):
    probabilities = load_probabilities(folder)
    return colorcode_probabilities(probabilities)
