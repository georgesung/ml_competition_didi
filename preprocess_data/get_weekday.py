'''
Given absolute time, return weekday as an integer between 0 and 6

Arguments:
Absolute time (integer)

Returns:
Integer between 0 and 6 inclusive. Monday is 0, Sunday is 6.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import calendar


def get_weekday(absolute_time):
	# First determine the year, month, day
	# NOTE: Below code only works for Jan/Feb 2016, it's not scalable but OK for this competition
	year = 2016
	month = absolute_time // (144*31) + 1
	day = (absolute_time - (144*31) * (month - 1)) // 144 + 1

	return calendar.weekday(year, month, day)