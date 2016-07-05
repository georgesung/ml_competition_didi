'''
Generate final csv result to submit
Reads y_test.npy as the result vector
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


def gen_test_result(test_data_path, test_result_file):
	# Read my raw results
	y_test = np.load(test_result_file)

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

	for t in range(num_timeslots):
		for d in range(NUM_DISTRICTS):
			idx = NUM_DISTRICTS * t + d

			print('%d,%s,%.6f' % (d+1, raw_to_predict[t], y_test[idx]))


if __name__ == '__main__':
	gen_test_result('../data/test_set_2/', 'y_test.npy')

