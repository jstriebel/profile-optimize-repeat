import numpy as np
import h5py


# Change: optimized numpy: dtypes, argmax re-use, del probabilities, rgb unrolled


def load_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        return f["probabilities"][:]


def colorcode_probabilities(probabilities):
    # Use uint8 to reference classes (we only have 4):
    predicted_classes = np.argmax(probabilities, axis=2).astype(np.uint8)
    # Do no re-compute max, use argmax result to index:
    max_prob = np.take_along_axis(probabilities, predicted_classes[..., None], axis=2).squeeze()
    # Use float32 to save memory:
    max_prob = max_prob.astype(np.float32)
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    colored_probabilities = class_colors[predicted_classes]
    # unroll RGB loop:
    colored_probabilities[:, :, 0] = colored_probabilities[:, :, 0] * max_prob
    colored_probabilities[:, :, 1] = colored_probabilities[:, :, 1] * max_prob
    colored_probabilities[:, :, 2] = colored_probabilities[:, :, 2] * max_prob
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    probabilities = load_probabilities(folder)
    return colorcode_probabilities(probabilities)
