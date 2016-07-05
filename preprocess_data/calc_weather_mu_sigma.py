'''
Given path to training weather data (directory), return mean and std-dev of temperature and PM2.5
Note the weather data covers all districts

Arguments:
1) Directory path to training weather data

Returns a tuple of (temp_mu, temp_sigma, pm25_mu, pm25_sigma)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

def calc_weather_mu_sigma(training_weather_data_path):
	# Lists to store individual temperature and PM2.5 levels
	temps = []
	pm25s = []

	# Make sure training data path exists
	assert os.path.exists(training_weather_data_path), 'calc_weather_mu_sigma.py: Training data directory does not exist'

	# Parse each weather file in training weather data
	for filename in os.listdir(training_weather_data_path):
		filename_full = os.path.join(training_weather_data_path, filename)

		# Parse individual weather data file
		with open(filename_full) as f:
			for line in f:
				line_vals = line.split()

				# Append to temperature and PM2.5 lists
				temps.append(float(line_vals[3]))
				pm25s.append(float(line_vals[4]))

	# Find mean and std-dev of temperature and PM2.5
	temp_mu    = np.mean(temps)
	temp_sigma = np.std(temps)
	pm25_mu    = np.mean(pm25s)
	pm25_sigma = np.std(pm25s)

	return (temp_mu, temp_sigma, pm25_mu, pm25_sigma)


if __name__ == '__main__':
	print(calc_weather_mu_sigma('../data/training_data/weather_data'))
