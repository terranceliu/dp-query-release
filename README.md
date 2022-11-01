# Setup

Our codebase currently supports Python **3.7**. We recommend that you create a separate virtual or Conda environment.

Install PyTorch **1.12.0** (other versions of PyTorch have not yet been tested) by following these 
[instructions](https://pytorch.org/get-started/previous-versions/) according to your system specifications. 
For example, the following command installs PyTorch with CUDA 11.6 support on Linux via pip.
````
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
````

Clone this repository and install the source files (via setuptools)
````
git clone git@github.com:terranceliu/dp-query-release.git
cd dp-query-release-priv
pip install -e .
````

# Data

Data can be preprocessed using the DataPreprocessor class found in from data_preprocessor.py. We provide an example of how to use this class in
````
python examples/data_preprocessing/preprocess_adult.py
````

# Execution

The examples directory provides examples of how to use this repository (more documentation to come). We provide the following script that provides an example of set of arguments that can be used to run GEM on the ADULT dataset.
````
./examples/scripts/gen_nn/run_adult.sh
````

# Acknowledgements

Portions of our code is adapted from

1) https://github.com/sdv-dev/CTGAN
2) https://github.com/ryan112358/private-pgm


[//]: # (conda create -n dp-query-release python=3.7)