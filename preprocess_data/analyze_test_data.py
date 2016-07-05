'''
Stand-alone script to analyze test data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from map_district_hash_id import map_district_hash_id
from parse_timestamp import parse_timestamp

# Number of districts
from settings import NUM_DISTRICTS
from settings import TRAIN_ABSOLUTE_TIME_RANGE
#from settings import TEST_ABSOLUTE_TIME_RANGE

# Raw text strings of timeslots to predict, and absolute time integer version
# Absolute time is number of 10-minute timeslots away from 2016-01-01 00:00:00
raw_to_predict = []
to_predict = []

# Populate above arrays based on Di-Tech provided read_me_2.txt
with open('../data/test_set_2/read_me_2.txt') as f:
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

print('raw_to_predict:\n%s' % raw_to_predict)
print('to_predict:\n%s' % to_predict)
print('TEST_ABSOLUTE_TIME_RANGE = (%d, %d)' % (to_predict[0], to_predict[-1]))

# Find the timeslots for all available traffic data in test set
traffic_timeslots = []
traffic_data_path = '../data/test_set_2/traffic_data'
for filename in os.listdir(traffic_data_path):
	filename_full = os.path.join(traffic_data_path, filename)

	# Parse individual traffic data file
	with open(filename_full) as f:
		for line in f:
			line_vals = line.split()

			_, __, absolute_time = parse_timestamp(line_vals[5] + ' ' + line_vals[6])

			traffic_timeslots.append(absolute_time)

# For each timeslot to predict, find the nearest previous traffic data timeslot, and report the delta
deltas = []
for i in range(len(to_predict)):
	nearest_traffic_t = -1
	for traffic_t in traffic_timeslots:
		if traffic_t < to_predict[i] and traffic_t > nearest_traffic_t:
			nearest_traffic_t = traffic_t

	deltas.append((raw_to_predict[i], to_predict[i] - nearest_traffic_t))

print('deltas:\n%s' % deltas)


