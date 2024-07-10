import numpy as np
import h5py


# Change: chunked reading and processing


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
    predicted_classes = np.argmax(probabilities, axis=2).astype(np.uint8)
    max_prob = np.take_along_axis(probabilities, predicted_classes[..., None], axis=2).squeeze()
    del probabilities
    max_prob = max_prob.astype(np.float32)
    class_colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]], dtype=np.uint8)
    colored_probabilities = class_colors[predicted_classes]
    colored_probabilities[:, :, 0] = colored_probabilities[:, :, 0] * max_prob
    colored_probabilities[:, :, 1] = colored_probabilities[:, :, 1] * max_prob
    colored_probabilities[:, :, 2] = colored_probabilities[:, :, 2] * max_prob
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        x_len, y_len = f["probabilities"].shape[:2]
    colored_probabilities = np.zeros((x_len, y_len, 3), dtype=np.uint8)
    for chunk_slice, chunk in load_probabilities(folder):
        colored_probabilities[chunk_slice] = colorcode_probabilities(chunk)
    return colored_probabilities
