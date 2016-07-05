'''
Generate test input data
This input data will be used to produce the final output submitted to Di-Tech competition

Arguments:
Path to training data

Returns:
x: a matrix with shape (total_test_time * NUM_DISTRICTS, num_features)

When this file is run stand-alone, it will produce file 'x_test.npy',
which stores the x matrix to disk, in current directory
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import calendar

from parse_timestamp import parse_timestamp
from gen_ds_gap import gen_ds_gap
from gen_poi_data import gen_poi_data
from get_weekday import get_weekday
from gen_traffic_data import gen_traffic_data

from settings import TRAIN_ABSOLUTE_TIME_RANGE
from settings import TEST_ABSOLUTE_TIME_RANGE
from settings import NUM_DISTRICTS
from settings import NUM_POI_CLASSES
from settings import TRAFFIC_TIME_DELAY


def gen_test_data(test_data_path, training_data_path):
	# Find out what timeslots we need to make predictions on
	raw_to_predict = []  # Raw text strings of timeslots to predict
	to_predict = []  # Absolute time (integer) of timeslots to predict

	# Populate above arrays based on Di-Tech provided read_me_2.txt
	with open(os.path.join(test_data_path, 'read_me_2.txt')) as f:
		line_num = 0
		for line in f:
			line_num += 1
			line = line.rstrip()

			# Skip first line
			if line_num == 1:
				continue

			raw_to_predict.append(line)

			# Create a dummy timestamp to get absolute time of the date
			line_vals = line.split('-')
			_, __, absolute_time = parse_timestamp(line_vals[0] + '-' + line_vals[1] + '-' + line_vals[2] + ' 00:00:00')

			# Add the timeslot value from read_me_2.txt to absolute time to get correct value for it
			absolute_time += int(line_vals[3])

			to_predict.append(absolute_time)

	# Number of timeslots to predict as determined by read_me_2.txt
	num_timeslots = len(to_predict)

	# Number of features, i.e. length of feature vector
	num_features = NUM_DISTRICTS + NUM_POI_CLASSES + 7 + 1 + 1

	# Create individual tensors needed to populate final matrix 'x'

	# One-hot encoding of district ID
	districts_repeated = []  # this will be [0, 1, ..., 65, 0, 1, ..., 65, ... (repated num_timeslots times)]
	for t in range(num_timeslots):
		for d in range(NUM_DISTRICTS):
			districts_repeated.append(d)
	districts_data = np.eye(NUM_DISTRICTS)[districts_repeated]  # this is the 1-hot districts vector repeated appropriately
	print(districts_data.shape)

	# POI class data for district
	poi_data = gen_poi_data(test_data_path) # normalized POI data with shape (NUM_DISTRICTS, NUM_POI_CLASSES)
	poi_data = np.tile(poi_data, (num_timeslots, 1))  # Tile the above "downwards"
	print(poi_data.shape)

	# One-hot encoding of weekday
	first_t = True  # to create initial numpy array, then concatenate to it (cannot create correct empty array)
	for t in to_predict:
		weekday = get_weekday(t)  # returns integer between 0 and 6, inclusive
		weekday = np.eye(7)[weekday]  # one-hot encoding of weekday
		weekday = np.tile(weekday, (NUM_DISTRICTS, 1))  # tile vertically NUM_DISTRICTS times

		if first_t:
			weekday_data = weekday
			first_t = False
		else:
			weekday_data = np.concatenate((weekday_data, weekday), axis=0)
	print(weekday_data.shape)

	# Daily timeslot value (raw value between 1 - 144, normalized according to uniform distribution)
	mu = np.mean([x+1 for x in range(144)])
	sigma = np.std([x+1 for x in range(144)])
	daily_timeslot = []
	for t in to_predict:
		for d in range(NUM_DISTRICTS):
			daily_timeslot.append([(t%144 - mu) / sigma])
	daily_timeslot = np.array(daily_timeslot)
	print(daily_timeslot.shape)

	# Get required traffic data TRAFFIC_TIME_DELAY timeslots prior to timeslot-to-predict
	raw_traffic_data = gen_traffic_data(training_data_path, test_data_path, is_training=False)
	traffic_data = np.full((num_timeslots * NUM_DISTRICTS, 1), np.nan)
	for i in range(len(to_predict)):
		for d in range(NUM_DISTRICTS):
			district_id = d + 1
			row_index_raw = (to_predict[i] - TEST_ABSOLUTE_TIME_RANGE[0] - TRAFFIC_TIME_DELAY) * NUM_DISTRICTS + (district_id - 1)
			row_index = i * NUM_DISTRICTS + (district_id - 1)
			traffic_data[row_index][0] = raw_traffic_data[row_index_raw][0]
	print(traffic_data.shape)
	
	# Concatenate all individual feature tensors into one big input tensor
	# This is the matrix we want to return
	x = np.concatenate((districts_data, poi_data, weekday_data, daily_timeslot, traffic_data), axis=1)
	print(x.shape)

	return x


if __name__ == '__main__':
	x_test = gen_test_data('../data/test_set_2/', '../data/training_data/')

	np.save('x_test.npy', x_test)
	