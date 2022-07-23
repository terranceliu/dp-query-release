# Setup

We provide instructions for installing this codebase using Anaconda (which we recommend for installing PyTorch).

Create a virtual environment with Python 3.7 (other versions of Python may work but have not been tested.)
````
conda create -n dp-query-release python=3.7
````

Activate your environment (remember to do this before running any code)
````
conda activate dp-query-release
````

Install [PyTorch](https://pytorch.org/get-started/locally/) 1.12.0 (other versions of PyTorch have not yet been tested) by following the instructions according to your system specifications.
````
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
````

Clone this repository and install the source files (via setuptools)
````
git clone git@github.com:terranceliu/dp-query-release-priv.git
cd dp-query-release-priv
pip install -e .
````

Please also add the path to this repository to your PYTHONPATH to make importing modules simpler.

# Data

Data can be preprocessed using the DataPreprocessor() found in from data_preprocessor.py. We provide an example of how to use this class in
````
python examples/data_preprocessing/preprocess_adult.py
````

# Execution

The examples directory provides examples of how to use this repository (more documentation to come). We provide the following script that provides an example of set of arguments that can be used to run GEM on the ADULT dataset.
````
./examples/scripts/run_gem_adult.sh
````

# Acknowledgements

Portions of our code is adapted from

1) https://github.com/sdv-dev/CTGAN
2) https://github.com/ryan112358/private-pgm

