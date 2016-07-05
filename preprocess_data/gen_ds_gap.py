'''
Generate demand-supply gap vector, indexed by timeslot

Arguments:
Directory path to data

Returns:
Matrix of shape (total_absolute_time * NUM_DISTRICTS, 1), containing the demand-supply gap at each timeslot for each district
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


def gen_ds_gap(data_path, is_training=True):
	assert os.path.exists(data_path), 'gen_ds_gap.py: Data directory does not exist'

	# Determine the number of timeslots our data encompasses
	if is_training:
		absolute_time_range = TRAIN_ABSOLUTE_TIME_RANGE
	else:
		print('gen_order_data.py ERROR: FIXME! Need to support test phase!!!')
		return
	total_absolute_time = absolute_time_range[1] - absolute_time_range[0] + 1

	# Obtain our district hash_to_id mapping
	hash_to_id, _ = map_district_hash_id(os.path.join(data_path, 'cluster_map/cluster_map'))

	# Create and initialize ds_gap to zeros
	ds_gap = np.zeros((total_absolute_time * NUM_DISTRICTS))

	# Parse all the order_data files and update ds_gap accordingly
	order_data_path = os.path.join(data_path, 'order_data')
	for filename in os.listdir(order_data_path):
		filename_full = os.path.join(order_data_path, filename)

		# Parse individual order data file
		with open(filename_full) as f:
			for line in f:
				line_vals = line.split()

				# Only care about this entry if order was not filled (i.e. there's a demand-supply gap)
				if line_vals[1] == 'NULL':
					# Determine the absolute time
					_, __, absolute_time = parse_timestamp(line_vals[6] + ' ' + line_vals[7])

					# Determine district_id
					# TODO: Confirm if we're predicting the ds_gap for *start* district. I'm assuming so.
					district_hash = line_vals[3]
					district_id = hash_to_id[district_hash]

					# Update ds_gap
					index = (absolute_time - absolute_time_range[0]) * NUM_DISTRICTS + (district_id - 1)
					ds_gap[index] += 1

	# Convert the 1D vector into 2D column vector, and return it
	return np.array([ds_gap]).transpose()


if __name__ == '__main__':
	ds_gap = gen_ds_gap('../data/training_data/')

	print(ds_gap)
