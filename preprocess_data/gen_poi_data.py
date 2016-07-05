'''
Generates normalized POI data for each district, for all first-level POI categories

Argument:
Path to data

Returns:
2D list of shape (NUM_DISTRICTS, NUM_POI_CLASSES)
The value of each POI class is normalized to the mean and std-dev of the POI class
i.e. normalized down the column
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from map_district_hash_id import map_district_hash_id

from settings import NUM_DISTRICTS
from settings import NUM_POI_CLASSES


def gen_poi_data(data_path):
	assert os.path.exists(data_path), 'gen_poi_data.py: Data directory does not exist'

	# Obtain the district hash_to_id mapping
	hash_to_id, _ = map_district_hash_id(os.path.join(data_path, 'cluster_map/cluster_map'))

	# Create raw_poi_data 2D tensor and initialize to 0
	# This will hold the un-normalized raw values
	raw_poi_data = np.zeros((NUM_DISTRICTS, NUM_POI_CLASSES))

	with open(os.path.join(data_path, 'poi_data/poi_data')) as f:
		for line in f:
			line_vals = line.split()

			# Determine district_id
			district_hash = line_vals[0]
			district_id = hash_to_id[district_hash]

			# For the rest of the line, parse each "section" separately
			for section in line_vals[1:]:
				poi_class = int(section.split(':')[0].split('#')[0])
				poi_val   = int(section.split(':')[1])

				# Update raw_poi_data
				raw_poi_data[district_id-1][poi_class-1] += poi_val

	# Find mean and std-dev for all POI classes in raw_poi_data
	poi_mu    = np.mean(raw_poi_data, axis=0)
	poi_sigma = np.std(raw_poi_data, axis=0)

	# Normalize the POI data
	# numpy handles the vector broadcasting
	poi_data = (raw_poi_data - poi_mu) / poi_sigma

	return poi_data


if __name__ == '__main__':
	poi_data = gen_poi_data('../data/training_data/')

	print(poi_data)
