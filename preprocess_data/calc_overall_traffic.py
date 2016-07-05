'''
Given raw traffic data, calculate & return overall traffic value
Includes parameters/constants to tune calculation of overall traffic value
Argument must be a list of length 4: [lvl1_traffic, lvl2_traffic, lvl3_traffic, lvl4_traffic]

Returns overall traffic value
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Overall traffic value = TRAFFIC_SCORE_OFFSET + 1 * lvl1_traffic + 2 * lvl2_traffic + 3 * lvl3_traffic + 4 * lvl4_traffic
from settings import TRAFFIC_SCORE_OFFSET


def calc_overall_traffic(raw_traffic):
	overall_traffic = TRAFFIC_SCORE_OFFSET + raw_traffic[0] + 2 * raw_traffic[1] + 3 * raw_traffic[2] + 4 * raw_traffic[3]

	return overall_traffic