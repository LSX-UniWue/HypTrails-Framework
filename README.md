# Code 2023 Hydras

## Description

This repository is focused on automating the derivation of interpretable hypotheses from various features. 
We  focus is on automating hypothesis generation from state-level features, considering factors like popularity and similarity. 
Various methods, including feature scaling and similarity measures, are implemented to create interpretable hypotheses. 
Artificial data will be used to develop algorithms for this purpose.

## Usage

Run `main.py`.  
Set the `synthetic` attribute to `True` to use synthetic data, otherwise set it to `False` to use bibliometric data. 

### Dataset

To create the bibliometric data, use the `create_dataset.py` script under `Code/extract_dblp_dataset`.