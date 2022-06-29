# Setup

Requires Python3. To install the necesssary packages, please run:
````
pip install -r requirements.txt
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

We adapt code from

1) https://github.com/sdv-dev/CTGAN
2) https://github.com/ryan112358/private-pgm

