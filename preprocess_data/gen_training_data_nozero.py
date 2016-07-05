'''
Generate training data, but remove all data points where the demand-supply gap is 0
Doing this because the MAPE metric does not penalize incorrect predictions
when the true demand-supply gap is 0.
I.e. If we predict DS gap of 1000 when true DS gap is 0, no penalty.

Arguments:
Path to training data

Returns:
x: matrix with shape (n, num_features)
y: matrix with shape (n, 1)

When this file is run stand-alone, it will produce files 'x_train_nozero.npy' and 'y_train_nozero.npy',
which stores the x and y matrices to disk, in current directory
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from gen_training_data import gen_training_data


def gen_training_data_nozero(training_data_path):
	# Get full training data
	x_train, y_train = gen_training_data(training_data_path)

	# Remove all rows in x_train and y_train where y_train[row][0] is 0
	delete_rows = []
	for row in range(y_train.shape[0]):
		if y_train[row][0] == 0.:
			delete_rows.append(row)

	x_train_nozero = np.delete(x_train, delete_rows, axis=0)
	y_train_nozero = np.delete(y_train, delete_rows, axis=0)

	# DEBUG
	print(x_train.shape)
	print(y_train.shape)
	print(x_train_nozero.shape)
	print(y_train_nozero.shape)

	return (x_train_nozero, y_train_nozero)


if __name__ == '__main__':
	x_train_nozero, y_train_nozero = gen_training_data_nozero('../data/training_data')

	np.save('x_train_nozero.npy', x_train_nozero)
	np.save('y_train_nozero.npy', y_train_nozero)
