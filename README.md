# tda-exploration
A learning project on the topic of topological data analysis (TDA)

t10k-images-idx3-ubyte - MNIST test images in IDX format

t10k-labels-idx1-ubyte - MNIST test images labels in IDX format

train-images-idx3-ubyte - MNIST training images in IDX format

train-labels-idx1-ubyte - MNIST training images labels in IDX format

mnistextr.py - contains function that converts IDX binary data into ndarray format

toppipe.py - chooses 40000 images from the training set, proceeds to extract topological features and labels for these images as well as for the images in the test set, saves the results in the files train_features.csv, train_labels.csv, test_features.csv, test_labels.csv, unless these files already exist.

Instead of running toppipe.py, it is recommended to download the topological data from this link:

https://drive.google.com/file/d/1FePK6hjhjInv6C97aE3euHymcLKF6G16/view?usp=sharing
