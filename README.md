# Lab 4: Naive Bayes

**CSCI 360: Introduction to Artificial Intelligence**

## Introduction
In this lab you will be implementing **Naive Bayes** on a Breast Cancer data set. The algorithm uses a provided training set. You are expected to estimate posterior probabilities using training data.

In this lab you will be implementing the algorithm, but first you will
have to clean the data. The data is found in the [`data.npy`](./data.npy).

All the code you write should be in [`lab4.py`](./lab4.py) and will be
under functions `preprocess_data`, `naive_bayes`and`cross_validation`(optional). It is
important you don't change the parameters. You are provided with a
utility file and a test file. The utility file has functions provided
that will compute the `load_data`.

It also contains the names of the features in the dataset in the order that they appear in the columns.

You are allowed to use `numpy` which is outlined by [`requirements.txt`](./requirements.txt)




## Test File:
The test file will try to use the `preprocess_data`, `naive_bayes`and`cross_validation` as they are outline in the lab4 PDF.

The test file uses `load_data` to pull a tuple from the data.
