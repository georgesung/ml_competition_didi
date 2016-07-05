'''
Generates training data, input matrix x and label vector y
Feature vector is a concatenation of:
	One-hot encoding of district ID
	POI class data for district
	One-hot encoding of weekday
	Daily timeslot value (raw value between 1 - 144, normalized according to uniform distribution)
	Traffic data shifted by TRAFFIC_TIME_DELAY

Note TRAFFIC_TIME_DELAY is how many timeslots earlier is the traffic data available,
for the timeslots to be predicted in the test set

Arguments:
Path to training data

Returns:
x: matrix with shape (total_absolute_time * NUM_DISTRICTS - TRAFFIC_TIME_DELAY, num_features)
y: matrix with shape (total_absolute_time * NUM_DISTRICTS - TRAFFIC_TIME_DELAY, 1)

When this file is run stand-alone, it will produce files 'x_train.npy' and 'y_train.npy',
which stores the x and y matrices to disk, in current directory
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import calendar

from gen_ds_gap import gen_ds_gap
from gen_poi_data import gen_poi_data
from get_weekday import get_weekday
from gen_traffic_data import gen_traffic_data

from settings import TRAIN_ABSOLUTE_TIME_RANGE
from settings import NUM_DISTRICTS
from settings import NUM_POI_CLASSES
from settings import TRAFFIC_TIME_DELAY


def gen_training_data(training_data_path):
	# Determine the number of timeslots our data encompasses
	total_absolute_time = TRAIN_ABSOLUTE_TIME_RANGE[1] - TRAIN_ABSOLUTE_TIME_RANGE[0] + 1

	# Number of features, i.e. length of feature vector
	num_features = NUM_DISTRICTS + NUM_POI_CLASSES + 7 + 1 + 1

	# Create temporary x and y tensors
	# We will remove rows/data-points where we have nan's later on
	x_temp = np.full((total_absolute_time * NUM_DISTRICTS, num_features), np.nan)
	y_temp = gen_ds_gap(training_data_path)

	# Create the individual tensors needed to populate x_temp

	# One-hot encoding of district ID
	districts_repeated = []  # this will be [0, 1, ..., 65, 0, 1, ..., 65, ... (repated total_absolute_time times)]
	for t in range(total_absolute_time):
		for d in range(NUM_DISTRICTS):
			districts_repeated.append(d)
	districts_data = np.eye(NUM_DISTRICTS)[districts_repeated]  # this is the 1-hot districts vector repeated appropriately
	print(districts_data.shape)
	print(np.count_nonzero(np.isnan(districts_data)))

	# POI class data for district
	poi_data = gen_poi_data(training_data_path) # normalized POI data with shape (NUM_DISTRICTS, NUM_POI_CLASSES)
	poi_data = np.tile(poi_data, (total_absolute_time, 1))  # Tile the above "downwards"
	print(poi_data.shape)
	print(np.count_nonzero(np.isnan(poi_data)))

	# One-hot encoding of weekday
	for t in range(TRAIN_ABSOLUTE_TIME_RANGE[0], TRAIN_ABSOLUTE_TIME_RANGE[1] + 1):
		weekday = get_weekday(t)  # returns integer between 0 and 6, inclusive
		weekday = np.eye(7)[weekday]  # one-hot encoding of weekday
		weekday = np.tile(weekday, (NUM_DISTRICTS, 1))  # tile vertically NUM_DISTRICTS times

		if t == TRAIN_ABSOLUTE_TIME_RANGE[0]:
			weekday_data = weekday
		else:
			weekday_data = np.concatenate((weekday_data, weekday), axis=0)
	print(weekday_data.shape)
	print(np.count_nonzero(np.isnan(weekday_data)))

	# Daily timeslot value (raw value between 1 - 144, normalized according to uniform distribution)
	# The issue is that daily time is cyclical, e.g. timeslot 1 is very close to timeslot 144,
	# but the feature below does not capture that. Future enhancement.
	mu = np.mean([x+1 for x in range(144)])
	sigma = np.std([x+1 for x in range(144)])
	daily_timeslot = []
	for t in range(TRAIN_ABSOLUTE_TIME_RANGE[0], TRAIN_ABSOLUTE_TIME_RANGE[1] + 1):
		for d in range(NUM_DISTRICTS):
			daily_timeslot.append([(t%144 - mu) / sigma])
	daily_timeslot = np.array(daily_timeslot)
	print(daily_timeslot.shape)
	print(np.count_nonzero(np.isnan(daily_timeslot)))

	# Traffic data shifted by TRAFFIC_TIME_DELAY
	traffic_data = gen_traffic_data(training_data_path, training_data_path)  # matrix size (total_absolute_time * NUM_DISTRICTS) x 1
	remove_idx = [x for x in range((total_absolute_time - 2) * NUM_DISTRICTS, total_absolute_time * NUM_DISTRICTS)]
	traffic_data = np.delete(traffic_data, remove_idx, axis=0)  # delete data for last 2 timeslots
	nan_data = np.full((2 * NUM_DISTRICTS, 1), np.nan)
	traffic_data = np.concatenate((nan_data, traffic_data), axis=0)
	print(traffic_data.shape)
	print(np.count_nonzero(np.isnan(traffic_data)))  # should have 132 NaNs b/c we removed 2 timeslots

	# Concatenate all individual feature tensors into one big input tensor
	x_temp = np.concatenate((districts_data, poi_data, weekday_data, daily_timeslot, traffic_data), axis=1)
	print(x_temp.shape)

	# Temporarily concatenate x_temp and y_temp side-by-side (i.e. along axis 1)
	# This allows us to remove rows with nan's, for both x and y
	xy_temp = np.concatenate((x_temp, y_temp), axis=1)

	# Remove all rows with nan's
	xy = xy_temp[~np.isnan(xy_temp).any(axis=1)]
	print(xy.shape)

	x, y = np.split(xy, [num_features], axis=1)

	print(x.shape)
	print(y.shape)

	return (x, y)


if __name__ == '__main__':
	x_train, y_train = gen_training_data('../data/training_data')

	np.save('x_train.npy', x_train)
	np.save('y_train.npy', y_train)
