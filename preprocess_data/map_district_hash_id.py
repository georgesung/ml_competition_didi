'''
Create internal bi-directional dictionary (two dicts) to map district hash to/from district ID
Takes absolute or relative filepath to data/training_data/cluster_map/cluster_map

Returns (references to) the two dictionaries: hash->id and id->hash (in that order)

Note id is stored as an int, hash is a string
Note district_id is 1-indexed
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

def map_district_hash_id(file_path):
	# Make sure the file exists
	assert os.path.exists(file_path), 'map_district_hash_id.py: File does not exist'

	# Create dicts to map district hash->id and id->hash
	hash_to_id = {}
	id_to_hash = {}

	# Open the cluster_map file and parse it line-by-line
	with open(file_path) as f:
		for line in f:
			d_hash, d_id = line.split()
			d_id = int(d_id)  # make sure district_id is an integer

			hash_to_id[d_hash] = d_id
			id_to_hash[d_id] = d_hash

	return hash_to_id, id_to_hash


if __name__ == '__main__':
	hash_to_id, id_to_hash = map_district_hash_id('../data/training_data/cluster_map/cluster_map')
	print(hash_to_id)
	print(id_to_hash)
