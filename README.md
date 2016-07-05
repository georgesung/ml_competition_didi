# DiDi Research Machine Learning Algorithm Competition Entry

This is my entry into DiDi Research's machine learning algorithm competition (http://research.xiaojukeji.com/competition/main.action?competitionId=DiTech2016). Overall I ranked 460/775 in the competition (didi_comp_rank.png).

## Problem Statement

DiDi Research is a division of DiDi Chuxing, a Chinese ride-hailing company (https://en.wikipedia.org/wiki/Didi_Chuxing). The goal of this competition is to predict the demand-supply gap, i.e. the number of unfilled ride requests at a given time in a given physical district. For full details about this competition, please refer to http://research.xiaojukeji.com/competition/main.action?competitionId=DiTech2016

## Data

The provided data includes information about:
  * Order details (can be used to calculate demand-supply gap)
  * Points-of-interest (POI) for each district (e.g. entertainment, shopping, sports, etc.)
  * Traffic time-series information for each district
  * Weather time-series information for each district

DiDi provides the training data containing all the informationc categories above, in addition to test data withouth demand-supply gap information, used to calculate the final score and rankings.

The full data description and download link are available at http://research.xiaojukeji.com/competition/detail.action?competitionId=DiTech2016

## My Approach
I pre-processed the time-invariant data to feed into a fully-connected neural network, to make predictions on the demand-supply gap. The demand-supply gap is a continuous variable, therefore I approached this as a regression problem.

Side note:
Initially, I thought of using a recurrent neural network (RNN) to predict the demand-supply gap given the prior time-series data, as well as time-invariant data. However, given the timestamps of the data points in the test data set, I could not construct a continuous time-series using the test data. Given more time and a better understanding of time-series prediction and RNNs, a future enhancement would be to leverage the time-series data and possibly use RNNs as basis of prediction.

### Data Pre-Processing
The data was pre-processed using python and numpy. The data I pre-processed were:
  * Points-of-interest (POI) data for each district, represented as a tensor with shape (NUM_DISTRICTS, NUM_POI_CLASSES)
    - Each of the NUM_POI_CLASSES "columns" represents how many POI instances are in each district, e.g. how many entertainment venues in this district, how many sports venues, etc.
  * Date information, to create 2 feature tensors:
    - Weekday (Mon/Tue/etc.) one-hot encoding, a tensor with shape (NUM_DATA_POINTS, 7)
    - Daily 10-minute timeslot value, from 1-144 inclusive. A tensor with shape (NUM_DATA_POINTS, 144)
  * Traffic data, shifted backwards in time by 2 timeslots (20 minutes). A tensor with shape (NUM_DATA_POINTS, 1)
    - The traffic data was shifted backwards by 2 timeslots because the test data set requires a prediction of the demand-supply gap given traffic data from 20 minutes prior
    - I created my own metric to measure the overall traffic based on the raw traffic data provided. More details are available in 'preprocess_data/calc_overall_traffic.py'. The overall traffic score is one float number.

All the data above was standardized (subtract mean and divide by standard deviation) as necessary.

Note I did not make use of the weather data, because weather data was missing for many timeslots.

I also pre-processed the order information to calculate the demand-supply gap for each district in all applicable timeslots.

Then, I tiled and concatenated all of the above as necessary, to create the training data feature tensor of shape (NUM_DATA_POINTS, NUM_FEATURES), and label tensor of shape (NUM_DATA_POINTS, 1).
Further, I removed all data points where the demand-supply gap is 0, because a demand-supply gap of 0 is not factored into the final score of the competition.

Finally, I ran principle component analysis (PCA) on the data to reduce dimensionality, since I had many one-hot feature vectors.

### Training and Test Result Generation
The creation of the neural network and training process was performed using TensorFlow.

Due to time contraints, I stuck with a 6 layer fully-connected neural network, with the following layer sizes: [128, 128, 128, 128, 128, 256]. ReLU non-linearity and dropout were added for all 6 layers. Since this is a regression problem, the final neural network output (prediction) is the sum of all outputs from the final layer (the 256-wide layer).

For regularization, L2 regularization in addition to dropout were used.

For the TensorFlow code used to construct the neural network, see the code between "#BEGIN model construction" and "#END model construction" in 'training/tf_nn_final_out.py'.

#### Hyper-Parameter Search
I used random uniform search to search across hyper-parameter value combinations for the following hyper-parameters:
  * L2 regularization strength
  * Dropout keep-probability
  * Learning rate (for Adam optimizer)

The code for the hyper-parameter search is at 'training/tf_nn_sweep.py'.

#### Model Training and Test Result Generation
After finding the optimal hyper-parameters, I used them to perform the final training and test result generation. The code to do this is located at 'training/tf_nn_final_out.py'.

## Running the Code
### Dependencies
  * Python 2 or 3
  * NumPy
  * scikit-learn
  * TensorFlow

### Steps to Run
  * git clone <this_repository>
  * cd ml_competition_didi
  * (download the training and test data into 'data/' directory)
  * cd preprocess_data
  * python gen_training_data_nozero.py
    - Generates training data, stores the feature tensor and labels into 'x_train_nozero.npy' and 'y_train_nozero.npy'
  * python gen_test_data.py
    - Generates test set, stores the feature tensor into 'x_test.npy'
  * python run_pca.py
    - Runs PCA on both training and test data
    - Somehow sklearn's PCA does not work in the TensorFlow environment, at least on my end. Running the PCA step on a new terminal with the vanilla Anaconda environment worked for me.
  * cd ../training
  * OPTIONAL: python tf_nn_sweep.py
    - Runs the hyper-parameter search
  * python tf_nn_final_out.py
    - Performs training on the model, generates final test results
  * cd ../preprocess_data
  * python gen_test_result.py > final.csv
	- 'final.csv' will be the csv file submitted to the competition