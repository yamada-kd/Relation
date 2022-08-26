# Relation

This repository includes source codes to generate neural networks with "Attention", "Linear Attention" or "Relation", which were benchmarked on two different problems (evaluation performance on a NLP benchmark dataset and computation time on a dataset including longer sequences). The benchmark results were on "[Relation is an option for processing context information](https://www.biorxiv.org/content/10.1101/2022.04.14.488336v1)."

For both problems (GLUE and (IMDb and Reuters Newswire Classification)), same architectures were used. The first layer is an input layer; the second layer is a layer of "Attention", "Linear Attention" or "Relation"; the third layer is an output layer.

1. Input layer
2. "Attention", "Linear Attention" or "Relation" layer
3. Output layer

There are two Python codes in the repository. The first one is for GLUE benchmark to evaluate the ability to process sequence data. The second one is for IMDb or Reuters Newswire Classification dataset to evaluate computation time of each method.

## glue.py
Generating model with "Attention", "Linear Attention" or "Relation" or without any of them to process data in GLUE benchmark dataset. GLUE dataset is not provided in the repository.
The directory "result" includes the benchmark results of each method on GLUE.

## imdbReuters.py
Calculates computation time of model with "Attention", "Linear Attention" or "Relation" or without any of them for IMDb or Reuters Newswire Classification dataset, provided by TensorFlow.

## wikitext.py and wikitext.sample.txt
Calculates computation time of model with "Attention", "Linear Attention" or "Relation" or without any of them for WikiText-103 dataset. WikiText-103 includes pairs of a header and sentences in an instance. The task in the study is the binary classification, where each method classify whether the input sentence belongs to the secondary level of headers (= = [string] = =) or the other level of headers; the secondary level of headers is positive flag (1) and the other level of headers is negative flag (0). The file "wikitext.sample.txt" includes sample instances.
