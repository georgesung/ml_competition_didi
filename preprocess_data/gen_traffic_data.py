'''
Given path to raw data, parse and process raw traffic data to produce normalized traffic values

Arguments:
Path to training data

Returns:
A matrix of size (total_absolute_time * NUM_DISTRICTS) x 1
[[some_traffic_value_0],   <- time 0, district 1
 [some_traffic_value_1],   <- time 0, district 2
 ...
 [some_traffic_value_i],   <- time 1, district 1
 [some_traffic_value_i+1]  <- time 1, district 2
 ...
]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from map_district_hash_id import map_district_hash_id
from parse_timestamp import parse_timestamp
from calc_overall_traffic import calc_overall_traffic
from calc_traffic_mu_sigma import calc_traffic_mu_sigma

# Number of districts
from settings import NUM_DISTRICTS
from settings import TRAIN_ABSOLUTE_TIME_RANGE
from settings import TEST_ABSOLUTE_TIME_RANGE
from settings import TRAFFIC_TIME_DELAY


def gen_traffic_data(training_data_path, data_path, is_training=True):
	# Make sure training_data_path and data_path exists
	assert os.path.exists(training_data_path), 'gen_traffic_data.py: Training data directory does not exist'
	assert os.path.exists(data_path), 'gen_traffic_data.py: Data directory does not exist'

	# Determine the number of timeslots our data encompasses
	if is_training:
		absolute_time_range = TRAIN_ABSOLUTE_TIME_RANGE
	else:
		# For test data we want traffic data TRAFFIC_TIME_DELAY timeslots prior to timeslot-to-predict
		temp = TEST_ABSOLUTE_TIME_RANGE
		absolute_time_range = (temp[0] - TRAFFIC_TIME_DELAY, temp[1])
	total_absolute_time = absolute_time_range[1] - absolute_time_range[0] + 1

	# Collect useful information
	hash_to_id, _ = map_district_hash_id(os.path.join(training_data_path, 'cluster_map/cluster_map'))
	traffic_mu_sigma = calc_traffic_mu_sigma(os.path.join(training_data_path, 'traffic_data'), hash_to_id)

	# Initialize my 2D tensor to be returned
	# Initialize with 0, since we are missing significant traffic data and cannot throw away too much other data
	# Also we are normalizing the traffic data to 0-mean, so have 0 data should be OK
	traffic_data = np.zeros((total_absolute_time * NUM_DISTRICTS, 1))

	# Go through all raw traffic data and update traffic_data accordingly
	traffic_data_path = os.path.join(data_path, 'traffic_data')
	for filename in os.listdir(traffic_data_path):
		filename_full = os.path.join(traffic_data_path, filename)

		# Parse individual traffic data file
		with open(filename_full) as f:
			for line in f:
				line_vals = line.split()

				# Determine the absolute time
				_, __, absolute_time = parse_timestamp(line_vals[5] + ' ' + line_vals[6])

				# Calculate overall traffic value
				district_id = hash_to_id[line_vals[0]]
				raw_traffic = [int(x.split(':')[1]) for x in line_vals[1:5]]
				overall_traffic = calc_overall_traffic(raw_traffic)

				# Normalize overall traffic value according the mu and sigma of the overall traffic value of the district
				mu, sigma = traffic_mu_sigma[district_id-1]
				overall_traffic_norm = (overall_traffic - mu) / sigma

				# Update traffic_data
				row_index = (absolute_time - absolute_time_range[0]) * NUM_DISTRICTS + (district_id - 1)
				traffic_data[row_index][0] = overall_traffic_norm

	return traffic_data

if __name__ == '__main__':
	traffic_data = gen_traffic_data('../data/training_data', '../data/training_data', is_training=True)
	print(traffic_data[:132])

	num_nan = np.count_nonzero(np.isnan(traffic_data))
	print('# of NaNs: %d' % num_nan)
	percent_nan = num_nan / ((TRAIN_ABSOLUTE_TIME_RANGE[1] + 1) * NUM_DISTRICTS)
	print('Percentage of NaNs: %.2f' % (percent_nan * 100))
