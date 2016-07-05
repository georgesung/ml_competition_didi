'''
Create the fully-connected neural network
Run training on the training set using data from '../preprocess_data/'
Run inference on the test set, and save the inferred labels to file '../preprocess_data/y_test.npy'
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from scipy import stats
import time
import math
import random

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 400, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_integer('num_features', 13, 'Number of features in input data.')

# NOTE: train_size + test_size == total_data_size = 102624 -- make sure that's the case
flags.DEFINE_integer('train_size', 86000, 'Number of data points in training set.')
flags.DEFINE_integer('test_size', 16624, 'Number of data points in test set.')


def inverse_boxcox(y, ld):
	# Credit: http://stackoverflow.com/a/36210153
	if ld == 0:
		return(np.exp(y))
	else:
		return(np.exp(np.log(ld*y+1)/ld))


def run_training(l2_scale_in, keep_prob_in, learning_rate_in):
	# Load my data
	x_all = np.load('../preprocess_data/x_train_pca.npy')
	y_all = np.squeeze(np.load('../preprocess_data/y_train_nozero.npy'))	# y is a column vector (nx1 matrix), make it 1D vector

	# Train-test split
	# Note we have 102624 training data points
	# We want a training data size that's easily divisible into batches, so let train_size = 86000
	# Thus we have test_size = 16624, or roughly 16% of data is test set
	x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size=FLAGS.train_size, random_state=1)

	# Execute box-cox transform for y_train only
	y_train_bc, boxcox_lambda = stats.boxcox(y_train)

	# Our targets/labels are positively skewed (skewed towards 1), like a geometric/exponential distribution
	# Split y_all into 2 bins based on threshold, and randomly pick w/o replacement from each bin to create batches
	threshold = 5
	num_batches = 2000

	small_y_idx = np.where(y_train <= threshold)[0]
	big_y_idx   = np.where(y_train > threshold)[0]

	assert small_y_idx.shape[0] >= FLAGS.batch_size, 'ERROR: Bin size smaller than small_y batch size'
	assert big_y_idx.shape[0] >= FLAGS.batch_size, 'ERROR: Bin size smaller than big_y batch size'

	x_train_batches = []
	y_train_batches = []

	for i in range(num_batches):
		assert FLAGS.batch_size%2 == 0, 'ERROR: FLAGS.batch_size must be even number'
		small_idx = np.random.choice(small_y_idx, size=int(FLAGS.batch_size/2))
		big_idx   = np.random.choice(big_y_idx, size=int(FLAGS.batch_size/2))

		x_train_batch = x_train[np.concatenate((small_idx, big_idx))]
		y_train_batch = y_train_bc[np.concatenate((small_idx, big_idx))]

		x_train_batches.append(x_train_batch)
		y_train_batches.append(y_train_batch)

	# For the test data, we can only perform inference on batches of FLAGS.batch_size
	# So split the test data into batches of batch_size
	# The last batch will (likely) be a partial batch, so fill it with dummy data
	assert (x_test.shape[0] % FLAGS.batch_size) != 0, 'ERROR: No support for perfectly sized test set vs. batch size!'
	num_batches_test = int(x_test.shape[0] // FLAGS.batch_size) + 1
	last_batch_idx = (num_batches_test - 1) * FLAGS.batch_size
	# These are the "full" batches
	x_test_batches = np.split(x_test[:last_batch_idx], num_batches_test - 1)
	y_test_batches = np.split(y_test[:last_batch_idx], num_batches_test - 1)
	# For the partial batches, figure out how much padding is needed
	num_pad = FLAGS.batch_size - (x_test.shape[0] % FLAGS.batch_size)
	# Create the last batch by concatenating the remaining test data, and some dummy data for padding
	x_test_last_batch = np.concatenate((x_test[last_batch_idx:], x_test[:num_pad]), axis=0)
	y_test_last_batch = np.concatenate((y_test[last_batch_idx:], y_test[:num_pad]), axis=0)
	# Finally, append this result to my list of test batches
	x_test_batches.append(x_test_last_batch)
	y_test_batches.append(y_test_last_batch)

	# Do the same for x_final for final output
	x_final = np.load('../preprocess_data/x_test_pca.npy')

	# For the final data, we can only perform inference on batches of FLAGS.batch_size
	# So split the final data into batches of batch_size
	# The last batch will (likely) be a partial batch, so fill it with dummy data
	assert (x_final.shape[0] % FLAGS.batch_size) != 0, 'ERROR: No support for perfectly sized final set vs. batch size!'
	num_batches_final = int(x_final.shape[0] // FLAGS.batch_size) + 1
	last_batch_idx_final = (num_batches_final - 1) * FLAGS.batch_size
	# These are the "full" batches
	x_final_batches = np.split(x_final[:last_batch_idx_final], num_batches_final - 1)
	# For the partial batches, figure out how much padding is needed
	num_pad_final = FLAGS.batch_size - (x_final.shape[0] % FLAGS.batch_size)
	# Create the last batch by concatenating the remaining final data, and some dummy data for padding
	x_final_last_batch = np.concatenate((x_final[last_batch_idx_final:], x_final[:num_pad_final]), axis=0)
	# Finally, append this result to my list of final batches
	x_final_batches.append(x_final_last_batch)

	# Tell TensorFlow that the model will be built into the default Graph.
	with tf.Graph().as_default():
		# Generate placeholders for the features and labels.
		features_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.num_features))
		labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size))

		# Keep probability for drop-out, will be fed in
		keep_prob = tf.placeholder(tf.float32)

		# Build a Graph that computes predictions from the inference model, as chosen during import.
		#pred = model(features_placeholder, keep_prob)  # NOTE: Model is parameterized within the model itself, can change later
		# I'm gonna slap the entire model over here, to more easily add L2 loss terms
		reg_loss = tf.Variable(0., name='reg_loss')

		# BEGIN model construction
		NUM_FEATURES = 13
		NUM_HIDDEN_LAYERS = 6
		HIDDEN_SIZES = [128, 128, 128, 128, 128, 256]

		w0   = tf.Variable(
			tf.truncated_normal([NUM_FEATURES, HIDDEN_SIZES[0]], stddev=1.0 / math.sqrt(float(NUM_FEATURES))))
		b0   = tf.Variable(tf.zeros([HIDDEN_SIZES[0]]))
		act0 = tf.nn.relu(tf.matmul(features_placeholder, w0) + b0, name='relu')
		drp0 = tf.nn.dropout(act0, keep_prob)

		w1   = tf.Variable(
			tf.truncated_normal([HIDDEN_SIZES[0], HIDDEN_SIZES[1]], stddev=1.0 / math.sqrt(float(HIDDEN_SIZES[0]))))
		b1   = tf.Variable(tf.zeros([HIDDEN_SIZES[1]]))
		act1 = tf.nn.relu(tf.matmul(drp0, w1) + b1, name='relu')
		drp1 = tf.nn.dropout(act1, keep_prob)

		w2   = tf.Variable(
			tf.truncated_normal([HIDDEN_SIZES[1], HIDDEN_SIZES[2]], stddev=1.0 / math.sqrt(float(HIDDEN_SIZES[1]))))
		b2   = tf.Variable(tf.zeros([HIDDEN_SIZES[2]]))
		act2 = tf.nn.relu(tf.matmul(drp1, w2) + b2, name='relu')
		drp2 = tf.nn.dropout(act2, keep_prob)

		w3   = tf.Variable(
			tf.truncated_normal([HIDDEN_SIZES[2], HIDDEN_SIZES[3]], stddev=1.0 / math.sqrt(float(HIDDEN_SIZES[2]))))
		b3   = tf.Variable(tf.zeros([HIDDEN_SIZES[3]]))
		act3 = tf.nn.relu(tf.matmul(drp2, w3) + b3, name='relu')
		drp3 = tf.nn.dropout(act3, keep_prob)

		w4   = tf.Variable(
			tf.truncated_normal([HIDDEN_SIZES[3], HIDDEN_SIZES[4]], stddev=1.0 / math.sqrt(float(HIDDEN_SIZES[3]))))
		b4   = tf.Variable(tf.zeros([HIDDEN_SIZES[4]]))
		act4 = tf.nn.relu(tf.matmul(drp3, w4) + b4, name='relu')
		drp4 = tf.nn.dropout(act4, keep_prob)

		w5   = tf.Variable(
			tf.truncated_normal([HIDDEN_SIZES[4], HIDDEN_SIZES[5]], stddev=1.0 / math.sqrt(float(HIDDEN_SIZES[4]))))
		b5   = tf.Variable(tf.zeros([HIDDEN_SIZES[5]]))
		act5 = tf.nn.relu(tf.matmul(drp4, w5) + b5, name='relu')
		drp5 = tf.nn.dropout(act5, keep_prob)

		pred = tf.reduce_sum(drp5, 1)

		reg_loss_raw = tf.nn.l2_loss(w0) + tf.nn.l2_loss(b0) + \
			tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + \
			tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2) + \
			tf.nn.l2_loss(w3) + tf.nn.l2_loss(b3) + \
			tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4) + \
			tf.nn.l2_loss(w5) + tf.nn.l2_loss(b5)

		l2_scale = tf.constant(l2_scale_in)
		reg_loss = l2_scale * reg_loss_raw
		# END model construction
	
		# Add to the Graph the Ops for loss calculation (MAPE).
		# Assume all 0-values in labels have been removed, otherwise we will get div-by-0
		pred_p1 = pred + tf.constant(1.)
		labels_placeholder_p1 = tf.add(labels_placeholder, tf.constant(1.))
		loss_vec = tf.abs(tf.div(tf.sub(labels_placeholder_p1, pred_p1), labels_placeholder_p1))  # abs((labels - pred) / labels)
		train_loss = tf.reduce_mean(loss_vec, name='mape')
		#train_loss = tf.reduce_mean(tf.square(tf.sub(pred, labels_placeholder)), name='mse')

		# Add L2 loss to total loss
		loss = train_loss + reg_loss

		# Add to the Graph the Ops that calculate and apply gradients.
		# Add a scalar summary for the snapshot loss.
		#tf.scalar_summary(loss.op.name, loss)
		# Create the gradient descent optimizer with the given learning rate.
		#optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_in)

		# Create a variable to track the global step.
		global_step = tf.Variable(0, name='global_step', trainable=False)
		# Use the optimizer to apply the gradients that minimize the loss
		# (and also increment the global step counter) as a single training step.
		train_op = optimizer.minimize(loss, global_step=global_step)

		# Add the Op to compare pred to labels during evaluation.
		eval_correct = loss  # FIXME: Not necessary, but keeping it here temporarily

		# Build the summary operation based on the TF collection of Summaries.
		#summary_op = tf.merge_all_summaries()

		# Add the variable initializer Op.
		init = tf.initialize_all_variables()

		# Create a saver for writing training checkpoints.
		#saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		#summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		# And then after everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		# Start the training loop.
		for step in range(FLAGS.max_steps):
			start_time = time.time()

			# Fill a feed dictionary with the actual set of features and labels
			# for this particular training step.
			feed_dict = {
				features_placeholder: x_train_batches[step % num_batches],
				labels_placeholder:   y_train_batches[step % num_batches],
				keep_prob: keep_prob_in,
			}

			# Run one step of the model.  The return values are the activations
			# from the `train_op` (which is discarded) and the `loss` Op.  To
			# inspect the values of your Ops or variables, you may include them
			# in the list passed to sess.run() and the value tensors will be
			# returned in the tuple from the call.
			#_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			_, loss_value, inf, tl, l2l = sess.run([train_op, loss, pred, train_loss, reg_loss], feed_dict=feed_dict)  # DEBUG

			duration = time.time() - start_time

			# Write the summaries and print an overview fairly often.
			if step % 1000 == 0:
				# Print status to stdout.
				print('Step %d: loss = %.6f (%.3f sec) -- train_loss = %.6f, l2/reg_los = %.6f' % (step, loss_value, duration, tl, l2l))

			# At final step calculate test MAPE loss
			if (step + 1) == FLAGS.max_steps:
				# For the test data, perform model inference to calculate predictions.
				# Recall we had to split the test data into batches of size FLAGS.batch_size,
				# with some padding as well.
				# Also for test loss report both MSE loss and MAPE loss
				cumm_error = 0.
				for i in range(num_batches_test):
					# Run inference on current batch in test set
					inf = sess.run(pred, feed_dict={features_placeholder: x_test_batches[i], keep_prob: 1.0})
					inf = np.squeeze(inf)  # convert the returned column matrix into vector

					# Get the "real" prediction value
					inf = inverse_boxcox(inf, boxcox_lambda)

					# The final batch has padding, so only account for real data-points
					if i == num_batches_test - 1:
						idx = y_test.shape[0] % FLAGS.batch_size
						error = np.sum(np.absolute((y_test_batches[i][:idx] - inf[:idx]) / y_test_batches[i][:idx]))
					# Else, accumulate normalized absolute error for the entire batch
					else:
						error = np.sum(np.absolute((y_test_batches[i] - inf) / y_test_batches[i]))

					cumm_error += error

					if i % 150 == 0:
						print('TEST PHASE: i = %d, error = %.6f' % (i, error))
						print('inputs (first 2):\n%s' % x_test_batches[i][:2])
						print('labels:\n%s' % y_test_batches[i])
						print('predictions:\n%s' % inf)

				# Calculate MAPE value for test data (i.e. test loss), and report it
				test_mape = cumm_error / y_test.shape[0]
				print('Step %d: test loss = %.6f' % (step, test_mape))

		# Report final training loss
		print('Final training loss: %.6f' % loss_value)

		# Calcuate final result from x_test.npy and write the resulting np array to disk
		y_final = np.array([])
		for i in range(num_batches_final):
			# Run inference on current batch in final test set
			inf = sess.run(pred, feed_dict={features_placeholder: x_final_batches[i], keep_prob: 1.0})
			inf = np.squeeze(inf)  # convert the returned column matrix into vector

			# Get the "real" prediction value
			inf = inverse_boxcox(inf, boxcox_lambda)

			# The final batch has padding, so only account for real data-points
			if i == num_batches_final - 1:
				idx = x_final.shape[0] % FLAGS.batch_size
			# Else, accumulate normalized absolute error for the entire batch
			else:
				idx = FLAGS.batch_size

			y_final = np.concatenate((y_final, inf[:idx]))

		print('Writing y_final to file y_test.npy, the shape is %s' % y_final.shape)
		np.save('../preprocess_data/y_test.npy', y_final)


def main(_):
	#run_training(0.0001, 0.3, 0.0001)
	run_training(0.01, 0.3, 0.0001)


if __name__ == '__main__':
	tf.app.run()
