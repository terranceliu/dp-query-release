# Setup

Requires Python3. To install the necesssary packages, please run:
````
pip install -r requirements.txt
````

Please also add the path to this repository to your PYTHONPATH to make importing modules simpler.

# Data

Data can be preprocessed using functions from preprocess.py. We provide an example of how to use these functions:
````
python examples/data_preprocessing/preprocess_bank.py
````

We also provide a preprocessed version of the ADULT dataset.

# Execution

The examples directory provides examples of how to use this repository (more documentation to come). We provide the following script that provides an example of set of arguments that can be used to run GEM on the ADULT dataset.
````
/examples/scripts/run_gem_adult.sh
````

# Acknowledgements

We adapt code from

1) https://github.com/sdv-dev/CTGAN
2) https://github.com/ryan112358/private-pgm

