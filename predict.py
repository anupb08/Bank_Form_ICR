import argparse
import os
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import data_handler as dh


# HELPER FUNCTIONS
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def init_weights(shape):
    """Returns random initial weights"""
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    """Returns random initial biases"""
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    """Returns a 2d convolution operation with stride size 1 and padding SAME"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    """Returns a 2 by 2 pooling operation with padding SAME"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape, name="unspecified"):
    """Returns a convolutional layer with random weights and biases"""
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = init_weights(shape)
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([shape[3]])
        with tf.name_scope("Wx_plus_b"):
            preactive = conv2d(input_x, W) + b
            tf.summary.histogram("pre_activations", preactive)
    activations = tf.nn.relu(preactive, name="activation")
    tf.summary.histogram("activations", activations)
    return activations


def normal_full_layer(input_layer, size, act=tf.nn.relu, name="unspecified"):
    """Returns a full layer with random weights and biases"""
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        with tf.name_scope("weights"):
            W = init_weights([input_size, size])
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([size])
            variable_summaries(b)
        with tf.name_scope("Wx_plus_b"):
            preactive = tf.matmul(input_layer, W) + b
            tf.summary.histogram("pre_activations", preactive)
        activations = act(preactive, name="activation")
        tf.summary.histogram("activations", activations)
        return activations


def predict(images, model_type):
    tf.reset_default_graph()
    #X = np.array([single_image])  # Create array from image to fit shape of x (?,32,32,1)
    X= np.array(images)
    #print(X.shape)

    # DICT
    checks = {
        0 : "N",
        1 : "Y"
    }
    combines = {
        0 : "0",
        1 : "1",
        2 : "2",
        3 : "3",
        4 : "4",
        5 : "5",
        6 : "6",
        7 : "7",
        8 : "8",
        9 : "9",
        10: "A",
        11: "B",
        12: "C",
        13: "D",
        14: "E",
        15: "F",
        16: "G",
        17: "H",
        18: "I",
        19: "J",
        20: "K",
        21: "L",
        22: "M",
        23: "N",
        24: "O",
        25: "P",
        26: "Q",
        27: "R",
        28: "S",
        29: "T",
        30: "U",
        31: "V",
        32: "W",
        33: "X",
        34: "Y",
        35: "Z",
        36: " ",
    }

    alphabets = {
        0 : "A",
        1 : "B",
        2 : "C", #"i/I",
        3 : "D",
        4 : "E",
        5 : "F",
        6 : "G",
        7 : "H",
        8 : "I",
        9 : "J",
        10: "K",
        11: "L",
        12: "M", #"M/m",
        13: "N",
        14: "O",
        15: "P", #"x/X",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V", #"C/c",
        22: "W",
        23: "X",
        24: "Y", #"o/O",
        25: "Z",
        26: " ",
    }

    digits = {
        0 : "0",
        1 : "1",
        2 : "2",
        3 : "3",
        4 : "4",
        5 : "5",
        6 : "6",
        7 : "7",
        8 : "8",
        9 : "9",
        10: " ",
    }

    if model_type == 'digits':
        classes = digits
        model_dir = 'models_digits_256'
        filter1 = 16
        filter2 = 32
        n_fc = 256
        checkpoint =os.path.join(model_dir, "32x32_2conv_16_32_1norm_256.ckpt")
    elif model_type == 'alphabets':
        classes = alphabets
        model_dir = 'models_alphabets_UC2'
        filter1 = 16
        filter2 = 32
        n_fc = 512
        checkpoint =os.path.join(model_dir, "32x32_2conv_16_32_1norm_512.ckpt")
    elif model_type == 'checks':
        classes = checks
        model_dir = 'models_checks'
        filter1 = 4
        filter2 = 8
        n_fc = 16
        checkpoint =os.path.join(model_dir, "32x32_2conv_4_8_1norm_16.ckpt")
    else:
        classes = combines
        model_dir = 'models_alphabets_digits'
        filter1 = 16
        filter2 = 32
        n_fc = 1024
        checkpoint =os.path.join(model_dir, "32x32_2conv_16_32_1norm_1024.ckpt")
    
    #checkpoint =os.path.join(model_dir, "32x32_2conv_32_64_1norm_1024.ckpt")  # Model used for prediction, must have the same graph structure!

    # VARIABLES
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="x")  # Input, shape = ?x32x32x1
    y_true = tf.placeholder(tf.float32, shape=[None, len(classes)], name="y_true")  # Labels, shape = ?x47

    # MODEL
    # filter size=(4,4); channels=1; filters=16; shape=?x32x32x32
    convo_1 = convolutional_layer(x, shape=[4, 4, 1, filter1], name="Convolutional_1")
    convo_1_pooling = max_pool_2by2(convo_1)  # shape=?x16x16x32

    # filter size=(4,4); channels=16; filters=32; shape=?x16x16x64
    convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, filter1, filter2], name="Convolutional_2")
    convo_2_pooling = max_pool_2by2(convo_2)  # shape=?x8x8x64
    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * filter2])

    # filter size=(4,4); channels=32; filters=64; shape=?x8x8x32
    #convo_3 = convolutional_layer(convo_2_pooling, shape=[4, 4, 32, 64], name="Convolutional_3")
    #convo_3_pooling = max_pool_2by2(convo_3)  # shape=4x4x32
    #convo_3_flat = tf.reshape(convo_3_pooling, [-1, 4 * 4 * 64])  # Flatten convolutional layer

    full_layer_one = normal_full_layer(convo_2_flat, n_fc, tf.nn.relu, name="Normal_Layer_1")
    with tf.name_scope("dropout"):
        hold_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", hold_prob)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
    y_pred = normal_full_layer(full_one_dropout, len(classes), act=tf.identity,
                               name="Output_Layer")  # Layer with 47 neurons for one-hot encoding
    with tf.name_scope("cross_entropy"):
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))  # Calculate cross-entropy
    tf.summary.scalar("cross_entropy", cross_entropy)
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002)  # Optimizer
        train = optimizer.minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_predictions"):
            correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))  # use argmax to get the index
            # of the highest value in the prediction array and compare that with the true array to generate and array
            #  of the form [True,False,True]
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))  # Calculate percentage of correct
            # predictions
    tf.summary.scalar("accuracy", accuracy)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)  # Restore saved Variables
        predictions = sess.run(y_pred, feed_dict={x: X, hold_prob: 1})
        sorted_pred = sorted(predictions[0])
        #print(sorted_pred[-1], sorted_pred[-2])
        prob2 = (sorted_pred[-1] - sorted_pred[-2] )*len(sorted_pred)/sorted_pred[-1]
        min1 = min(sorted_pred)
        sorted_pred = sorted_pred + abs(min1)
        all_sum = sum(sorted_pred)
        sorted_pred = sorted_pred/all_sum
        max1 = (sorted_pred)[-1]
        max2 = (sorted_pred)[-2]
        prob = (max1 -max2)*max1*100*len(sorted_pred)
    #return (classes[predictions[1].argmax()], round(prob,1), round(prob2,1))
        outputs = []
        for predict in predictions:
            outputs.append(classes[predict.argmax()])
    return (outputs, round(prob,1), round(prob2,1))



def getPredictionResult(images, model_type):
    #single_image = dh.get_2d_array(image)
    imgs = []
    for image in images:
        im_color = cv2.resize(image, (32,32))  # Rescale the image
        im = np.zeros(shape=(32,32,1))  # Create an empty array with the final shape (32,32,1)
    #for i, x in enumerate(im_color):  # Fill the array
    #    for n, y in enumerate(x):  # Note: We cannot use cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY), because
    #        im[i][n][0] = (y[0] + y[0] + y[0]) / 3  # that will return an array with the shape (32,32), but we need
        for i in range(32):
            for j in range(32):
                im[i][j][0] = im_color[i][j]
        imgs.append(im)

    return predict(imgs, model_type)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Path to the image file")
    args = vars(ap.parse_args())
    print(args["image"])
    images = []
    for image in args["image"].split():
        single_image = cv2.imread(image,cv2.COLOR_BGR2GRAY)
        #single_image = dh.get_2d_array(image)
        images.append(single_image)
    #print("\nResult: \"" + predict(images,'combines') + "\".")
    print("\nResult: \"" + str(getPredictionResult(images, 'combines')) + "\".")
