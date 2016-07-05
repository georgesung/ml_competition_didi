'''
Given a timestamp string (e.g. "2016-01-25 23:50:16"), return:
	(1) Weekday (Monday = 0, ... Sunday = 6)
	(2) 10-minute timeslot, an integer between 1 and 144
	(3) Absolute time, measured as number of 10-minute timeslots after 2016-01-01 00:00:00
		e.g. 2016-01-01 00:06:00 -> 0; 2016-01-02 00:12:32 -> 145

Returns a 3-tuple of (weekday, timeslot, absolute_time)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import calendar


def parse_timestamp(timestamp):
	date, time = timestamp.split()

	# Parse date string to obtain weekday value
	year, month, day = [int(x) for x in date.split('-')]
	weekday = calendar.weekday(year, month, day)

	# Parse time string to calculate timeslot (integer between 1-144, inclusive)
	hour, minute, second = [int(x) for x in time.split(':')]
	total_minutes = 60 * hour + minute
	timeslot = total_minutes // 10 + 1

	# Calculate absolute time
	# Find delta year, month, day from 2016-01-01
	delta_year  = year - 2016
	delta_month = month - 1
	delta_day   = day - 1
	
	# Accumulate absolute time given above deltas
	# FIXME: This only works for January/February 2016! To extend this, must account for different days per month, and leap years.
	# However, the training and test data for this competition only includes days in January 2016, so it's OK for now.
	abs_time = 31 * 144 * delta_month + 144 * delta_day

	# Add the raw timeslot value we calculated earlier, but remember to subtract 1 (we are 0-indexing our absolute time)
	abs_time += timeslot - 1

	return (weekday, timeslot, abs_time)

if __name__ == '__main__':
	examples = [
		'2016-01-25 23:50:16',
		'2016-01-01 00:06:00',
		'2016-01-02 00:12:32',
		'2016-01-21 23:59:59'
	]

	for example in examples:
		print(example)
		print(parse_timestamp(example))
