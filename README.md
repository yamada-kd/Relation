# Relation

This repository includes source code to generate neural networks with "Attention", "Linear Attention" or "Relation", which were benchmarked on two different problems (evaluation performance on a NLP benchmark dataset and computation time on a dataset including longer sequences). The benchmark results were on "[Relation is an option for processing context information](https://www.biorxiv.org/content/10.1101/2022.04.14.488336v1)"

## glue.py
Generating model with "Attention", "Linear Attention" or "Relation" or without any of them to process data in GLUE benchmark dataset. GLUE dataset is not provided in the repository.
The directory "result" includes the benchmark results of each method on GLUE.

## imdbReuters.py
Calculates computation time of model with "Attention", "Linear Attention" or "Relation" or without any of them for IMDb or Reuters Newswire Classification dataset, provided by TensorFlow.
