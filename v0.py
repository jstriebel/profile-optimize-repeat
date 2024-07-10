import numpy as np


def load_probabilities(folder):
    arrays = [np.loadtxt(folder / f"{c}.csv") for c in range(4)]
    return np.stack(arrays, axis=-1)


def colorcode_probabilities(probabilities):
    class_colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]
    colored_probabilities = []
    for probabilities_row in probabilities:
        colored_probabilities_row = []
        for class_probabilities in probabilities_row:
            class_index = None
            max_prob = 0
            for i, prob in enumerate(class_probabilities):
                if prob > max_prob:
                    max_prob = prob
                    class_index = i
            colored_probability = [c * max_prob for c in class_colors[class_index]]
            colored_probabilities_row.append(colored_probability)
        colored_probabilities.append(colored_probabilities_row)
    return colored_probabilities


def load_and_colorcode_probabilities(folder):
    probabilities = load_probabilities(folder)
    return colorcode_probabilities(probabilities)
