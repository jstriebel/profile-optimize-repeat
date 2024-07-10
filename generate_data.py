import functools
from pathlib import Path
from shutil import rmtree

import h5py
import numpy as np
import skimage
from sklearn.ensemble import RandomForestClassifier

# This script is heavily based on the following scikit-image tutorial:
# https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html

SKIN = skimage.data.skin()

def train():
    img = SKIN[:900, :900]

    # Hardcoded training labels:
    training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
    training_labels[:130] = 1
    training_labels[:170, :400] = 1
    training_labels[600:900, 200:650] = 2
    training_labels[330:430, 210:320] = 3
    training_labels[260:340, 60:170] = 4
    training_labels[150:200, 720:860] = 4

    features_func = functools.partial(
        skimage.feature.multiscale_basic_features,
        intensity=True,
        edges=False,
        texture=True,
        sigma_min=1,
        sigma_max=16,
        channel_axis=-1,
    )
    features = features_func(img)
    classifier = RandomForestClassifier(
        n_estimators=50,
        n_jobs=-1,
        max_depth=10,
        max_samples=0.05
    )
    classifier = skimage.future.fit_segmenter(training_labels, features, classifier)
    return features_func, classifier

def predict_probabilities(image, features_func, classifier):
    features = features_func(image)
    if features.ndim > 2:
        features_flat = features.reshape((-1, features.shape[-1]))
    probabilities_flat = classifier.predict_proba(features_flat)
    probabilities = probabilities_flat.reshape(features.shape[:-1] + (-1,))
    return probabilities

def save_image_and_probabilities(folder, image, probabilities):
    rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    with h5py.File(folder / f"data.hdf5", "w") as f:
        f.create_dataset("image", data=image)
        f.create_dataset("probabilities", data=probabilities)
    for c in range(4):
        np.savetxt(folder / f"{c}.csv", probabilities[..., c])

def main():
    features_func, classifier = train()
    image = SKIN
    probabilities = predict_probabilities(image, features_func, classifier)
    for key, scale in {"s": 1, "m": 2, "l": 4, "xl": 6}.items():
        reps = (scale, scale, 1)
        image_tiled = np.tile(image, reps)
        probabilities_tiled = np.tile(probabilities, reps)
        save_image_and_probabilities(
            Path(f"data/{key}"),
            image_tiled,
            probabilities_tiled
        )

if __name__ == "__main__":
    main()
