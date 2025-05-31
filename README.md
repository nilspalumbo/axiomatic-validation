# Validating Mechanistic Interpretations: An Axiomatic Approach

## Installing Dependencies

First, make sure Conda and Node.js are installed. In the base environment, run:

```
conda install nb_conda_kernels
conda env create -f environment.yml
conda activate validating_mis
npm install vega-lite vega-cli canvas
```

Next, build [drat-trim](https://github.com/marijnheule/drat-trim):

```
git clone https://github.com/marijnheule/drat-trim.git
cd drat-trim
make
cp drat-trim ..
```


## Generating the training and analysis datasets

For the 2-SAT analysis, use the notebook `RandomCNFs.ipynb`; see "Generate Training Dataset" and "Generate Analysis Dataset." Datasets used for experiments are available [here](https://zenodo.org/records/11239102); download these to the "data" folder. The dataset used for training and testing the model is named cnf_tokens_1M.npy and the one used for our analysis is named cnf_tokens_100K.npy.

## Training the 2-SAT model

`python -u model.py --num_heads 1 4 --num_layers 2 --run_name layers_2_heads_1_4 --data_path ./data/cnf_tokens_1M.npy --train`

The trained model used for analysis and a log of the training process are in the "models/layers_2_heads_1_4" folder.

Running `modular_addition.ipynb` will download the trained model for the modular arithmetic case study.

## Code structure

`RandomCNFs.ipynb` contains the code to generate the 2-SAT datasets, `model.py` contains the model implementation, `plot.py` contains plotting code, `helpers.py` contains various utilities. The core code for the analysis of the 2-SAT model is in `interpretation.ipynb` and with the key implementation details in `interpretation.py`, including the decomposition of the model, our interpretation, and the alpha and gamma functions. `modular_addition_utils.py` defines the model, dataset, and various tools for the interpretation of the modular arithmetic model; `modular_addition.ipynb` contains the core code for the analysis of the modular arithmetic model with the key implementation details in `modular_addition_interpretation.py`.
`model.py`, `plot.py`, and `modular_addition_utils.py` are adapated from [Progress measures for grokking via mechanistic interpretability](https://github.com/mechanistic-interpretability-grokking/progress-measures-paper).

## Validating the mechanistic interpretations

The notebooks `interpretation.ipynb` and `modular_addition.ipynb` contain the code to validate the interpretations.

