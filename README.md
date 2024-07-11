# Profile, Optimize, Repeat

Code examples and [slides](slides%20profile%20optimize%20repeat.pdf) for the [EuroPython 2024](https://ep2024.europython.eu/session/profile-optimize-repeat-one-core-is-all-you-needtm) talk
> **Profile, Optimize, Repeat: One Core Is All You Need™** 

Speakers:
[Valentin Nieper](https://github.com/valentin-pinkau) & [Jonathan Striebel](https://github.com/jstriebel)


## Setup

Requirements:
* Python 3.11+
* Python venv
* CMake & C++ compiler (for the pypind11 extension)

```shell
# Clone the repo
git clone --recursive git@github.com:jstriebel/profile-optimize-repeat.git
# If you didn't clone with --recursive, add the submodules using
# git submodule update --init --recursive

cd profile-optimize-repeat
# Add and activate a Python venv
python3 -m venv .venv
. .venv/bin/activate

# Install requirements, also compiling the pybind11 extension
pip install -r requirements.txt

# Generate data used later, see "data" folder
python generate_data.py
```

The data generation is based on the scikit-image tutorial [Trainable segmentation using local features and random forests](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html)


## Usage

```shell
# Run the main program by specifying the versions to use:
python main.py v0 --show
# Besides the version to use, you can specify
# --show: whether to plot the images
# --data: the data folder, default is "data/s".

# Use scalene to profile the program. We start the profiler programatically,
# so it should be started with --off first. --cli gives nice output, otherwise
# the html version is shown in the browser
scalene --off --cli main.py --- v7 --data data/xl

# Add --cpu to only profile CPU time, avoiding memory profiling overhead
scalene --off --cli --cpu main.py --- v7 --data data/xl
```

## Changes between Versions

* v0: initial example code
* v1: change input from csv to hdf5
* v2: vectorize with numpy
* v3: optimized numpy
* v4: chunked reading and processing
* v5: numba jitting
* v6: custom C++ extension via pybind11
* v7: combine custom C++ extension with chunking


## Example Profiling Overview

The following table gives an overview of profiling runs on a single machine:

```
size s:
v0  3738 ms –  192.20 MB
v1  3071 ms –  230.03 MB
v2   123 ms –   88.26 MB
v3    49 ms –   48.43 MB
v4   102 ms –   46.61 MB
v5    42 ms –   38.24 MB
v6    32 ms –   38.24 MB
v7    76 ms –   34.27 MB

size m
v0 15235 ms –  787.85 MB
v1 10676 ms –  938.10 MB
v2   831 ms –  351.98 MB
v3   244 ms –  211.32 MB
v4   284 ms –   86.04 MB
v5    80 ms –  164.28 MB
v6    67 ms –  164.41 MB
v7   138 ms –   82.01 MB

size l:
v2  3283 ms – 1374.00 MB
v3  1155 ms –  844.08 MB
v4   999 ms –  137.25 MB
v5   364 ms –  656.49 MB
v6   317 ms –  656.63 MB
v7   431 ms –  156.28 MB

size xl:
v2  6570 ms – 3090.00 MB
v3  2851 ms – 1854.00 MB
v4  2220 ms –  199.56 MB
v5  1073 ms – 1442.00 MB
v6  1242 ms – 1442.00 MB
v7   924 ms –  191.61 MB
```

Details: Used separate profiling for CPU time and peak memory, Python 3.11.2 on Debian 12, ThinkPad X1 Carbon 5th Gen, Intel(R) Core(TM) i7-7500U
