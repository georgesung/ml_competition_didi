'''
Run PCA to reduce dimensionality of input data,
for both training and test data.

Since there are lots of one-hot encoded data, it's useful to run PCA on it

This script can be run stand-alone, and reads from
x_train_nozero.npy and x_test.npy

It will write to x_train_pca.npy and x_test_pca.npy

NOTE: Somehow sklearn's PCA does not work in TensorFlow environment,
make sure to run it in non-TF env
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.decomposition import PCA

x_train = np.load('x_train_nozero.npy')
x_test = np.load('x_test.npy')

'''
To see how to find n_components, run below

import matplotlib.pyplot as plt
pca = PCA(n_components=x_train.shape[1])
pca.fit(x_train)
plt.plot(pca.explained_variance_ratio_[:30], 'bo-')
plt.xlabel('PCA Dimension #')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. PCA Dimension #')

When plotted, a good value for n_components is 13
'''

pca = PCA(n_components=13)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

np.save('x_train_pca.npy', x_train_pca)
np.save('x_test_pca.npy', x_test_pca)
