'''
Given path to training traffic data (directory), return mean and std-dev of overall traffic value for all districts (do this in training)
This will read the training data directly and calculate the traffic mu and sigma for all districts

Arguments:
1) Directory path to training traffic data
2) District hash-to-id mapping dict

Returns a 2D list, where:
row_index = district_id - 1
column_index==0 -> mu
column_index==1 -> sigma
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from calc_overall_traffic import calc_overall_traffic

from settings import NUM_DISTRICTS


def calc_traffic_mu_sigma(training_traffic_data_path, hash_to_id):
	# Mapping of district_id to corresponding list of overall traffic values
	district_traffic = {}

	# Initialize the above
	for i in range(NUM_DISTRICTS):
		district_traffic[i+1] = []

	# Final matrix to return
	traffic_mu_sigma = np.zeros((NUM_DISTRICTS, 2))

	# Make sure training data path exists
	assert os.path.exists(training_traffic_data_path), 'calc_traffic_mu_sigma.py: Training data directory does not exist'

	# Parse each traffic file in training traffic data
	for filename in os.listdir(training_traffic_data_path):
		filename_full = os.path.join(training_traffic_data_path, filename)

		# Parse individual traffic data file
		with open(filename_full) as f:
			for line in f:
				line_vals = line.split()

				# Populate district_traffic dict
				district_id = hash_to_id[line_vals[0]]
				raw_traffic = [int(x.split(':')[1]) for x in line_vals[1:5]]
				overall_traffic = calc_overall_traffic(raw_traffic)

				district_traffic[district_id].append(overall_traffic)

	# For each district, calculate the mean and std-dev
	for district_id in district_traffic:
		overall_traffic = district_traffic[district_id]

		# Some district(s) have no traffic data, so just leave the mu and sigma at their initial values of 0.
		if len(overall_traffic) == 0:
			continue

		mu = np.mean(overall_traffic)
		sigma = np.std(overall_traffic)

		traffic_mu_sigma[district_id-1][0] = mu
		traffic_mu_sigma[district_id-1][1] = sigma

	return traffic_mu_sigma


if __name__ == '__main__':
	from map_district_hash_id import map_district_hash_id

	hash_to_id, id_to_hash = map_district_hash_id('../data/training_data/cluster_map/cluster_map')
	traffic_mu_sigma = calc_traffic_mu_sigma('../data/training_data/traffic_data/', hash_to_id)

	print(traffic_mu_sigma)
