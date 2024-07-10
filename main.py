from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

from scalene import scalene_profiler
import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer


@contextmanager
def profile():
    is_scalene_running = scalene_profiler.Scalene._Scalene__initialized
    if is_scalene_running:
        scalene_profiler.start()
    try:
        yield
    finally:
        if is_scalene_running:
            scalene_profiler.stop()


def load_image(folder):
    with h5py.File(folder / "data.hdf5", "r") as f:
        return f["image"][:]


def plot(image, visualization):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(22, 10))
        ax[0].set_title("Image")
        ax[0].imshow(image)
        ax[1].imshow(visualization)
        ax[1].set_title("Predictions")
        fig.tight_layout()
        plt.show()


def main(
    version: str,
    show: bool = False,
    data: Path = Path("data/s"),
):
    analysis_module = import_module(version)

    with profile():
        colored_probabilities = analysis_module.load_and_colorcode_probabilities(data)

    # Display the image
    if show:
        image = load_image(data)
        # convert explicitly to np.ndarray[uint8] for v0/v1
        colored_probabilities = np.array(colored_probabilities, dtype=np.uint8)
        plot(image, colored_probabilities)


if __name__ == '__main__':
    typer.run(main)
