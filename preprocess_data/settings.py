'''
Settings are stored here, to be imported by other scripts
'''

# Overall traffic value = TRAFFIC_SCORE_OFFSET + 1 * lvl1_traffic + 2 * lvl2_traffic + 3 * lvl3_traffic + 4 * lvl4_traffic
TRAFFIC_SCORE_OFFSET = 2  # this imples lvl4 traffic is 2 times worse than lvl 1 traffic: 2*(x + 1) = x + 4 -> x = 2

# Number of districts
NUM_DISTRICTS = 66

# Range of absolute time for training/test data (inclusive, in units of 10-minute timeslots)
# Training data time range is from '2016-01-01 00:00:00' to '2016-01-21 23:59:59'
TRAIN_ABSOLUTE_TIME_RANGE = (0, 3023)
TEST_ABSOLUTE_TIME_RANGE = (3214, 4462)

# Map the raw weather category integers to internal index value
NUM_WEATHER_CATEGORIES = 6
WEATHER_CATEGORY_MAP = {
	1: 0,
	2: 1,
	3: 2,
	4: 3,
	8: 4,
	9: 5
}

# Note POI raw indices run from 1 to 25 (1-indexed)
# Also, below is POI first-level categories
NUM_POI_CLASSES = 25

# Traffic data is TRAFFIC_TIME_DELAY timeslots away from timeslot to predict
TRAFFIC_TIME_DELAY = 2
