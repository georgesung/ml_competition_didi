'''
Given path to raw data, parse and process raw weather data to produce categorized and normalized weather data tensor

Returns a 2D tensor (matrix) of size (total_absolute_time * NUM_DISTRICTS) x 8
[[8-wide vector 0],   <- time 0, district 1
 [8-wide vector 1],   <- time 0, district 2
 ...
 [8-wide vector i],   <- time 1, district 1
 [8-wide vector i+1]  <- time 1, district 2
 ...
]

Our 8 "columns" consist of:
[1 column for normalized temperature, 1 column for normalized PM2.5, 6 columns for 1-hot encoding of weather category]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from parse_timestamp import parse_timestamp
from calc_weather_mu_sigma import calc_weather_mu_sigma

from settings import NUM_WEATHER_CATEGORIES  # NOTE: Should be 6 categories, code not tested to work otherwise
from settings import WEATHER_CATEGORY_MAP    # NOTE: ditto
from settings import NUM_DISTRICTS
from settings import TRAIN_ABSOLUTE_TIME_RANGE
#from settings import TEST_ABSOLUTE_TIME_RANGE

def gen_weather_data(training_data_path, data_path, is_training=True):
	# Make sure training_data_path and data_path exists
	assert os.path.exists(training_data_path), 'gen_weather_data.py: Training data directory does not exist'
	assert os.path.exists(data_path), 'gen_weather_data.py: Data directory does not exist'

	if is_training:
		absolute_time_range = TRAIN_ABSOLUTE_TIME_RANGE
	else:
		print('gen_weather_data.py ERROR: FIXME! Need to support test phase!!!')
		return

	# Determine temperature and PM2.5 mean and std-dev, to be used for data normalization later
	temp_mu, temp_sigma, pm25_mu, pm25_sigma = calc_weather_mu_sigma(os.path.join(training_data_path, 'weather_data'))

	# Initialize my 2D tensor to be returned. Initialize with nan value, to see where we are missing weather data
	# row_index = (absolute_time - absolute_time_range[0]) * NUM_DISTRICTS + (district_id - 1)
	total_absolute_time = absolute_time_range[1] - absolute_time_range[0] + 1
	weather_data = np.full((total_absolute_time * NUM_DISTRICTS, NUM_WEATHER_CATEGORIES + 2), np.nan)

	# Go through all raw weather data and update weather_data accordingly
	weather_data_path = os.path.join(data_path, 'weather_data')
	for filename in os.listdir(weather_data_path):
		filename_full = os.path.join(weather_data_path, filename)

		# Parse individual weather data file
		with open(filename_full) as f:
			for line in f:
				line_vals = line.split()

				# Determine the absolute time
				_, __, absolute_time = parse_timestamp(line_vals[0] + ' ' + line_vals[1])

				# Obtain raw weather category, temperature, and PM2.5 values
				weather_category, temp, pm25 = int(line_vals[2]), float(line_vals[3]), float(line_vals[4])

				# Create one-hot vector to represent weather category
				weather_category_internal = WEATHER_CATEGORY_MAP[weather_category]
				category_1hot = np.eye(NUM_WEATHER_CATEGORIES)[weather_category_internal]  # creates NUM_WEATHER_CATEGORIES-wide 1-hot vector

				# Calculate normalized temperature and PM2.5 values
				temp_norm = (temp - temp_mu) / temp_sigma
				pm25_norm = (pm25 - pm25_mu) / pm25_sigma

				# Update weather_data, for all districts in the given timeslot
				for d in range(NUM_DISTRICTS):
					# Recall district_id is 1-indexed
					district_id = d + 1
					row_index = (absolute_time - absolute_time_range[0]) * NUM_DISTRICTS + (district_id - 1)

					weather_data[row_index][0]  = temp_norm
					weather_data[row_index][1]  = pm25_norm
					weather_data[row_index][2:] = category_1hot

	return weather_data


if __name__ == '__main__':
	weather_data = gen_weather_data('../data/training_data', '../data/training_data', is_training=True)
	print(weather_data[:132])

	num_nan = np.count_nonzero(np.isnan(weather_data))
	percent_nan = num_nan / ((TRAIN_ABSOLUTE_TIME_RANGE[1] + 1) * NUM_DISTRICTS)
	print('Percentage of NaNs: %.2f' % (percent_nan * 100))

	# DEBUG: Find timeslot ranges where I am missing weather data
	DEBUG = True
	if DEBUG:
		num_rows, _ = weather_data.shape
		prev_row_is_nan = False

		ranges = []
		start = 0
		end = 0  # end is exlusive, to be pythonic (so in C/C++ end is actually end-1)

		for row in range(num_rows):
			if np.isnan(weather_data[row][0]):
				if not prev_row_is_nan:
					start = row

				prev_row_is_nan = True

			elif prev_row_is_nan:
				end = row
				ranges.append((start, end))

				prev_row_is_nan = False

		print('Ranges of contiguous nan:\n%s' % ranges)
		print('SORRY! Way too much missng data, dont think we can use this...')
