
# Python 2.7 compatible. Kindly do not run on Python 3 or above

# Machine Learning
# In this assignment, we had to try 3 different machine learning algorithms to identify the orientation of an image.
#
# ************************************************k-nearest neighbours***********************************************
# Part 1:
# kNN - Takes ~ 8-9 mins to run
# kNN being a lazy algorithm does not learn anything during training, but instead just stores the entire train data
# set. It then finds the distance b/w each test instance and each train distance and assigns the majority of the train
# class label of its "nearest" k neighbors.
# An optimal value of k varies from problem to problem and we need to find that by experimenting.
#
# Part 1:
# We have done the following for k-means:
# 1) As a part of training, we converted the  input image data from text files and stored them as numpy array
# 2) Then on the test data, we simply calculated the euclidean distance between each test instance and train instance
# and then return the majority of the train class label of its "nearest" k neighbors.
# 3) After experimentation we found k=50 to work the best with the provided data set.
#
# Note: Kindly see the report for accuracies over different values of k. And the confusion matrix when k = 50
#
# Design Decisions:
# Initially we used normal python list to store the train and test data and calculate the distance b/w them.
# But since the data was huge, it took almost ~1.5 hrs for knn to run on the full data set
# We then used np.array to store the train and test data and calculate the distance b/w them.
# This gave us a huge performance improvement, with the test time of ~8 mins on the full data set.
#
# Note: kNN takes ~8 mins to run on the provided data set. Time depends on the train and test data set size. Hence
# it might differ on other data sets.

# *************************************************************AdaBoost************************************************
# Part 2:
# AdaBoost - Takes ~ 15 seconds to run (Best Stump Count = 100)
# The basic idea of AdaBoost algorithm is to create a bunch of weak classifiers which in combination can make a strong
# decision.
# We have done the following for adaBoost:
# 1) We are taking 2 random points to decide whether a file has been classified correctly or incorrectly.
# 2) While training, for each orientation we are creating n stumps (n = stump_count) and storing their classifier weight
# and their 2 random points
# 3) While testing, we run these classifiers for each/per orientation. If a file a classified correctly, we add its
# alpha and subtract otherwise. Since this is being done for each orientation, we get 4 different alphas values, one
# for each orientation.
# 4) Now we compare for these alphas each orientation and consider the corresponding orientation of the highest alpha
# 5) After experimentation we found stump_count = 100 to work the best with the provided data set. We need to use a high
# stump value since we taking completely random pairs of points and not finding the best ones.
# The accuracy rises quickly at first, but then slows down once the stump size becomes greater than 25.
# It gets consistent and stops improving much after stump = 100.
# Also, since we are taking random points, the accuracies may vary (by ~2-3%) even for the same stump value.

#
# Note: Kindly see the report for accuracies over different values of stump_count. And the confusion matrix when
# stump_count = 100

# Note: We are using pandas.Dateframe just to display the confusion matrix, since it provides good alignment like that
# of a table. Pandas hasn't been used for any other purpose.

# *******************************************************Neural Network************************************************
# Part 3:
# Neural Network: Takes ~ 5-6 mins to run (Best Hidden Count = 25)
# You may try with smaller hidden count for better performance (Hidden Count = 10 ; T = ~200 seconds)

# We have done the following for Neural Network:
# 1) We have implemented a single layer neural network with 100 iterations
# 2) We are initializing the value of the learning rate (alpha) to be 0.9.
# Learning Rate (alpha) = 0.9 (I am increasing its power by 1 after every 10 iterations
# eg. After 20 iteration - alpha = (0.9)^2 ; after 30 iterations - alpha = (0.9)^3 and so on...)

# 3) We are using stochastic  gradient descent as our activation function
# 4) After experimentation we found hidden_count = 50 to work the best with the provided data set. We need to use a high
# stump value since we taking completely random pairs of points and not finding the best ones.
#
# Note: Kindly see the report for accuracies over different values of hidden_count. And the confusion matrix when
# hidden_count = 25

# Design Decisions:
# 1) Firstly, to improve performance, we tried not using all the 192 features for Neural Network. Instead, we were
# taking an average of 2 consecutive neighbors and storing as one value. This made our input feature vector for each
# train/test file to be of 96 instead of 192.
# This was done with an intuition that not much changes are captured in consecutive pixels and hence the information
# loss is minimal.
# This trick gives us a little performance improvement, but sacrificed ~3-5% of the accuracy.
# Hence, reverted the change back and used full set of features.
#
# 2) Also to avoid huge calculations, we are normalizing both the train and test input vectors.
# 3) Learning Rate (alpha) = 0.9 (Initially) (I am increasing its power by 1 after every 10 iterations
# eg. After 20 iteration - alpha = (0.9)^2 ; after 30 iterations - alpha = (0.9)^3 and so on...)

#
# Note: We are using pandas.Dataframe just to display the confusion matrix, since it provides good alignment like that
# of a table. Pandas hasn't been used for any other purpose.



import sys
import collections
from collections import defaultdict, Counter
import math as m
import time
import pickle
import random as r
from Queue import PriorityQueue
import pandas as pd
import numpy as np
from random import shuffle

# Global Variable

############################################Common Functions###################################################

# Writes a particular file file_name to the pickle file
def write_data_to_file(file_name, file_dict):
    print("Writing files to pickle...")
    pickle.dump(file_dict, open(file_name, "wb"))


# Reads the dict from the pickle files and stores in the global file_list
def read_data_from_file(file_name):
    print("Reading files from pickle...")
    file_dict = pickle.load(open(file_name, "rb"))
    return file_dict

# Converts the elements of list into int
def make_int_list(row):
    return [int(i) for i in row]


# Converts the elements of list into int and takes avg between 2 elements
def make_avg_int_list(row):
    row = [int(i) for i in row]
    row_avg = []
    for x in range(0,len(row), 2):
        avg = (row[x] + row[x+1])/2
        row_avg.append(avg)
    row_avg.append(1)
    return row_avg


# Creates a train dict from the given input file
def get_train_dict(train_file, algorithm):
    print "Reading train data..."
    try:
        with open(train_file) as f:
            full_data = [line.rstrip('\n').split() for line in f]
            if algorithm == 'adaboost':
                data_dict = defaultdict(dict)
                for row in full_data:
                    data_dict[row.pop(0)][int(row.pop(0))] = row
                return data_dict
            elif algorithm == 'nearest' or algorithm == 'best':
                train_fname_label_list = []
                train_grid_list = []
                for row in full_data:
                    train_fname_label_list.append((row.pop(0),int(row.pop(0))))
                    train_grid_list.append(make_int_list(row))
                train_grid = np.array(train_grid_list)
                return train_fname_label_list, train_grid
            elif algorithm == 'nnet':
                train_fname_label_list = []
                train_grid_list = []
                train_label_list = []
                orientation_list = get_orientation_list()
                shuffle(full_data)
                for row in full_data:
                    train_label_row = [0, 0, 0, 0]
                    train_label_row[orientation_list.index(int(row[1]))] = 1
                    train_label_list.append(train_label_row)
                    train_fname_label_list.append((row.pop(0), int(row.pop(0))))
                    # train_grid_list.append(make_avg_int_list(row))
                    row.append("1")
                    train_grid_list.append(make_int_list(row))
                train_grid = np.array(train_grid_list)
                train_label = np.array(train_label_list)
                return train_fname_label_list, train_label, train_grid
    except IOError:
        print("Cannot read from file:", train_file)
        return 0


# Creates a train dict from the given input file
def get_test_dict(test_file, algorithm):
    print "Reading test data..."
    try:
        with open(test_file) as f:
            full_data = [line.rstrip('\n').split() for line in f]
            if algorithm == 'adaboost':
                data_dict = defaultdict(tuple)
                # full_data = f.readlines()
                for row in full_data:
                    data_dict[row.pop(0)] = (int(row.pop(1)), make_int_list(row[1:]))
                return data_dict
            elif algorithm == 'nearest' or algorithm == 'best':
                test_fname_label_list = []
                test_grid_list = []
                for row in full_data:
                    # data_dict[row.pop(0)][int(row.pop(0))] = make_int_list(row[2:])
                    test_fname_label_list.append((row.pop(0),int(row.pop(0))))
                    test_grid_list.append(make_int_list(row))
                test_grid = np.array(test_grid_list)
                return test_fname_label_list, test_grid
            elif algorithm == 'nnet':
                test_fname_label_list = []
                test_grid_list = []
                test_label_list = []
                orientation_list = get_orientation_list()
                for row in full_data:
                    # data_dict[row.pop(0)][int(row.pop(0))] = make_int_list(row[2:])
                    test_label_row = [0, 0, 0, 0]
                    test_label_row[orientation_list.index(int(row[1]))] = 1
                    test_label_list.append(test_label_row)
                    test_fname_label_list.append((row.pop(0), int(row.pop(0))))
                    # test_grid_list.append(make_int_list(row.append(1)))
                    # test_grid_list.append(make_avg_int_list(row))
                    row.append("1")
                    test_grid_list.append(make_int_list(row))
                test_grid = np.array(test_grid_list)
                test_label = np.array(test_label_list)
                return test_fname_label_list, test_label, test_grid
    except IOError:
        print("Cannot read from file:", test_file)
        return 0


# Gives the list of possible orientations
def get_orientation_list():
    return [0, 90, 180, 270]


# Calculates accuracy based on the given confusion matrix
def calc_accuracy(test_correct_count, test_file_count):
    return (100.0 * test_correct_count) / test_file_count


# Creates confusion matrix dictionary
def create_confusion_matrix():
    confusion_matrix = collections.OrderedDict()
    # print "All topic prior", len(topic_priors)
    c_orientation_list = get_orientation_list()
    for orientation in c_orientation_list:
        confusion_matrix[orientation] = collections.OrderedDict()
        for other_orientations in c_orientation_list:
            confusion_matrix[orientation][other_orientations] = 0
        confusion_matrix[orientation]["Actual Count"] = 0
    # print "CM", confusion_matrix
    return confusion_matrix


# Print confusion matrix
def print_confusion_matrix(confusion_matrix):
    print "Confusion Matrix"
    print "\t Model Results"
    pc_orientation_list = get_orientation_list()
    pc_orientation_list.append("Actual Count")
    confusion_list = [[pc_orientation_list[y] for x in range(len(pc_orientation_list) + 1)] for y in
                      range(len(pc_orientation_list) - 1)]
    confusion_header = ["Actual\Model"] + pc_orientation_list
    for i in range(len(confusion_header) - 2):
        for j in range(1, len(confusion_header)):
            confusion_list[i][j] = confusion_matrix[pc_orientation_list[i]][confusion_header[j]]
    cm = pd.DataFrame(confusion_list, columns=confusion_header)
    print cm.to_string(index=False)
    print "\n"


# Prints the test file name with their orientation:
def write_test_output_file(test_output_dict, file_name):
    print("Saving Test Output...")
    f = open(file_name, "w")
    for t_file in test_output_dict:
        f.write(t_file + " " + str(test_output_dict[t_file][0]) + "\n")
        # f.write(t_file + " " + str(test_output_dict[t_file][0]) + " " + str(test_output_dict[t_file][1]) + "\n")

    f.write("\n")
    f.close()

############################################End Common Functions###################################################

############################################Adaboost Code###################################################

# Returns a list of n random numbers with the specified range
def generate_random_points(begin_range, end_range, total_points):
    col_list = r.sample(xrange(begin_range, end_range), total_points)
    return col_list


# Initializes the weights equally for all the files
def init_equal_weights(train_file_dict):
    weights_dict = defaultdict(float)
    total_files = len(train_file_dict)
    init_weight = 1.0/total_files
    for t_file in train_file_dict:
        weights_dict[t_file] = init_weight
    return weights_dict, init_weight, init_weight


# Adds the stump_model of each iteration to the trained_model
def update_train_model(trained_model, orientation, stump_no, alpha, col_list):
    trained_model[orientation][stump_no] = defaultdict(float)
    trained_model[orientation][stump_no]['alpha'] = alpha
    trained_model[orientation][stump_no]['col_list'] = col_list

# Checks whether the file is correctly classified or not
def decision_stump(col_list, file_coordinates_list):
    return True if int(file_coordinates_list[col_list[0]]) < int(file_coordinates_list[col_list[1]]) else False


# Calculates the alpha value for the given total error
def get_alpha(total_stump_error):
    if total_stump_error < 1.0:
        return 0.5 * m.log(((1.0-total_stump_error)/total_stump_error), 2)
        # return 1.0 * m.log(((1.0-total_stump_error)/total_stump_error), 2)
        # return -0.5 * m.log(total_stump_error,2)
    return 1.0


# Recalculate the weights for the next step based on the number of incorrectly classified examples
# Returns the new weights for correct and incorrect examples
def get_new_misclassified_weights(total_train_files, len_correct, len_incorrect):
    correct_weight = 0.5/len_correct
    incorrect_weight = 0.5/len_incorrect
    return correct_weight, incorrect_weight


# Recalculate the weights for the next step based on the number of incorrectly classified examples
# Returns the new weights for correct and incorrect examples
def get_new_misclassified_weights2(total_train_files, len_correct, len_incorrect, prev_correct_weight, prev_incorrect_weight):
    new_error = (1.0 * len_incorrect)/total_train_files
    new_correct_weight = (1.0 * prev_correct_weight * new_error)/(1-new_error)
    # Normalizing weights
    new_total = (len_correct * new_correct_weight) + (len_incorrect * prev_incorrect_weight)
    correct_weight = 1.0 * new_correct_weight/new_total
    incorrect_weight = 1.0 * prev_incorrect_weight/new_total
    return correct_weight, incorrect_weight


# Assign new weights to the train examples
def assign_new_weights(weights_dict, correct_dict, incorrect_dict, correct_weight, incorrect_weight):
    for t_file in correct_dict:
        weights_dict[t_file] = correct_weight
    for t_file2 in incorrect_dict:
        weights_dict[t_file2] = incorrect_weight


# Trains the AdaBoost algorithm for a given stump_count
def train_adaboost(train_file_dict, stump_count, orientation_list):
    trained_model = defaultdict(dict)
    for orientation in orientation_list:
        weights_dict, prev_correct_weight, prev_incorrect_weight = init_equal_weights(train_file_dict)
        for stump_no in range(1, stump_count+1):
            col_list = generate_random_points(0, 191, 2)
            correct_dict = defaultdict(float)
            incorrect_dict = defaultdict(float)
            for t_file in train_file_dict:
                correct_classification = decision_stump(col_list, train_file_dict[t_file][orientation])
                if correct_classification:
                    correct_dict[t_file] = weights_dict[t_file]
                else:
                    incorrect_dict[t_file] = weights_dict[t_file]
            # Calculating the sum of the weights of the misclassified examples
            total_stump_error = sum(incorrect_dict.values())
            # Calculating the alpha weight for this stump
            alpha = get_alpha(total_stump_error)
            # Storing the stump for this orientation in the trained_model
            update_train_model(trained_model, orientation, stump_no, alpha, col_list)
            # Recalculate the weights for the next step
            correct_weight, incorrect_weight = get_new_misclassified_weights(len(train_file_dict), len(correct_dict), len(incorrect_dict))
            # Assign new weights to the train examples
            assign_new_weights(weights_dict, correct_dict, incorrect_dict, correct_weight, incorrect_weight)
    return trained_model


# Tests the AdaBoost algorithm for a given trained model, test set and stump_count
# Returns a dict with test file name as key and its orientation as value
def test_adaboost(trained_model_dict, test_file_dict, stump_count, orientation_list):
    test_output_dict = defaultdict(int)
    test_correct_count = 0
    nb_confusion_matrix = create_confusion_matrix()
    for t_file in test_file_dict:
        max_alpha = (-100,-1)
        for orientation in orientation_list:
            o_alpha = 0
            for stump_no in range(1, stump_count+1):
                train_stump_alpha = trained_model_dict[orientation][stump_no]['alpha']
                col_list = trained_model_dict[orientation][stump_no]['col_list']
                t_file_pixel_list = test_file_dict[t_file][1]
                correct_classification = decision_stump(col_list, t_file_pixel_list)
                if correct_classification:
                    o_alpha += train_stump_alpha
                else:
                    o_alpha -= train_stump_alpha
            if o_alpha > max_alpha[0]:
                max_alpha = (o_alpha, orientation)
        # print "alpha", max_alpha[0]
        predicted_orientation = max_alpha[1]
        actual_orientation = test_file_dict[t_file][0]
        test_output_dict[t_file] = (predicted_orientation, actual_orientation)
        # Keeping the correct count for calculating accuracy
        if predicted_orientation == actual_orientation:
            test_correct_count += 1
        nb_confusion_matrix[actual_orientation][predicted_orientation] += 1
        nb_confusion_matrix[actual_orientation]["Actual Count"] += 1
    # Prints the confusion matrix
    print_confusion_matrix(nb_confusion_matrix)
    # Calculates the accuracy
    adaboost_accuracy = calc_accuracy(test_correct_count, len(test_file_dict))
    return test_output_dict, adaboost_accuracy

###########################################End of Adaboost Code##############################################

############################################kNN Code###################################################

# Calculates Euclidean Distance for 2 instances
def euclidean_distance(test_instance, train_instance):
    d = np.linalg.norm(test_instance-train_instance)
    return d

# Predicts the label based on the value of k
def predict_kNN_test_label(test_neighbours_queue, k_value):
    # print "Predicting test orientation using " + str(k_value) + " nearest neighbours..."
    train_label_count = Counter()
    for i in range(k_value):
        t_label = test_neighbours_queue.get()[1]
        train_label_count[t_label] += 1
    predicted_orientation = train_label_count.most_common(1)[0][0]
    return predicted_orientation


# Returns a priority queue of all the train neighbours for a given test exemplar where
# Priority key is the distance and value is the train label
def get_all_neighbours_queue(test_grid, train_grid, train_fname_label_list):
    test_neighbours_queue = PriorityQueue()
    for x in range(len(train_grid)):
        orientation = train_fname_label_list[x][1]
        e_dist = euclidean_distance(test_grid, train_grid[x])
        test_neighbours_queue.put((e_dist, orientation))
    return test_neighbours_queue


# Runs kNN classifier on the test data
def test_kNN(train_fname_label_list, train_grid, test_fname_label_list, test_grid, k_value, orientation_list):
    print "Testing kNN...Finding Euclidean Distance between Test and Train Data..."
    test_output_dict = defaultdict(tuple)
    test_file_no = 0
    test_correct_count = 0
    nb_confusion_matrix = create_confusion_matrix()
    total_test_files = len(test_grid)
    for x in range(len(test_grid)):
        test_file_no += 1
        print "Test file no: ", test_file_no, "/", total_test_files
        test_fname = test_fname_label_list[x][0]
        actual_orientation = test_fname_label_list[x][1]
        test_neighbours_queue = get_all_neighbours_queue(test_grid[x], train_grid, train_fname_label_list)
        predicated_orientation = predict_kNN_test_label(test_neighbours_queue, k_value)
        #test_output_dict[test_fname] = predicated_orientation
        test_output_dict[test_fname] = (predicated_orientation, actual_orientation)
        if predicated_orientation == actual_orientation:
            test_correct_count += 1
        nb_confusion_matrix[actual_orientation][predicated_orientation] += 1
        nb_confusion_matrix[actual_orientation]["Actual Count"] += 1
    # Prints the confusion matrix
    print_confusion_matrix(nb_confusion_matrix)
    # Calculates the accuracy
    knn_accuracy = calc_accuracy(test_correct_count, total_test_files)
    return test_output_dict, knn_accuracy

###########################################End of kNN Code##############################################

############################################Neural Net Code###################################################

# Returns the sigmoid of a input matrix
def sigmoid(z):
    return 1/(1+np.exp(-z))


# Returns a derivative of a matrix
def derivate_sigmoid(z):
    return z * (1-z)


# Updating the highest value in a2 to 1
def update_max_prob(a2):
    h_prob = max(a2)
    a_updated = []
    label_index = -1
    #for item in a2:
    for x in range(len(a2)):
        if h_prob == a2[x]:
            label_index = x
            a_updated.append(1)
        else:
            a_updated.append(0)
    a2 = np.array(a_updated)
    return a2, label_index


def train_nnet(train_fname_label_list, train_label, train_grid, hidden_count, orientation_list):
    print "Training Neural Net with Hidden nodes count =", str(hidden_count)
    print "Train Size: ", len(train_grid)
    nnet_model = defaultdict(int)
    input_layer_size = len(train_grid[0])
    hidden_layer_size = hidden_count
    output_layer_size = len(train_label[0])
    alpha = 0.9
    o_alpha = 0.9
    i_full = train_grid
    i_full = i_full/np.linalg.norm(i_full)
    np.random.seed(1)
    # Randomly initializing the weights
    # W1 = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
    # W2 = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1
    W1 = np.random.randn(input_layer_size, hidden_layer_size)
    W2 = np.random.randn(hidden_layer_size, output_layer_size)
    # W1 = 2 * np.random.random((Decimal(input_layer_size), Decimal(hidden_layer_size)))-1
    # W2 = 2 * np.random.random((Decimal(hidden_layer_size), Decimal(output_layer_size)))-1
    iterations = 101
    count = 1
    for iter in range(1, iterations):
        # Updating alpha (learning rate) after every 10 iterations
        if iter%10 == 0:
            alpha = m.pow(o_alpha,count)
            count += 1
        print "Iteration:", iter, "/", iterations-1
        for n in range(len(i_full)):
            label_index = -1
            i1 = i_full[n]
            # Feed Forward
            z1 = np.dot(i1, W1)
            a1 = sigmoid(z1)
            z2 = np.dot(a1, W2)
            a2 = sigmoid(z2)
            y = train_label[n]
            # Updating the highest value in a to 1
            if iter == iterations-1:
                a2, label_index = update_max_prob(a2)
            # Output error
            error_a2 = y - a2
            delta_a2 = derivate_sigmoid(a2) * error_a2
            error_a1 = delta_a2.dot(W2.T)
            delta_a1 = derivate_sigmoid(a1) * error_a1

            i1 = i1.reshape(i1.shape[0],1)
            delta_a1 = delta_a1.reshape((1, -1))
            W1 += alpha * i1.dot(delta_a1)

            a1 = a1.reshape(a1.shape[0],1)
            delta_a2 = delta_a2.reshape((1, -1))
            W2 += alpha * a1.dot(delta_a2)

    nnet_model["W1"] = W1
    nnet_model["W2"] = W2
    return nnet_model


def test_nnet(nnet_model, test_fname_label_list, test_label, test_grid, hidden_count, orientation_list):
    print "Testing Neural Net..."
    print "Test Size: ", len(test_grid)
    # nnet_model = read_data_from_file("nnet_model.p")
    test_output_dict = defaultdict(tuple)
    nb_confusion_matrix = create_confusion_matrix()
    test_correct_count = 0
    correct_count = 0
    i_full = test_grid
    i_full = i_full/np.linalg.norm(i_full)
    np.random.seed(1)
    W1 = nnet_model["W1"]
    W2 = nnet_model["W2"]
#    for m in range(1):
    for n in range(len(i_full)):
        label_index = -1
        i1 = i_full[n]
        # Feed Forward
        z1 = np.dot(i1, W1)
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2)
        a2 = sigmoid(z2)
        y = test_label[n]
        # Updating the highest value in a to 1
        a2, label_index = update_max_prob(a2)
        # Output error
        error_a2 = y - a2
        # For Accuracy
        test_fname = test_fname_label_list[n][0]
        actual_orientation = test_fname_label_list[n][1]
        predicted_orientation = orientation_list[label_index]
        test_output_dict[test_fname] = (predicted_orientation, actual_orientation)
        # if np.mean(np.abs(error_a2)) == 0.0:
        #     correct_count += 1
        if predicted_orientation == actual_orientation:
            test_correct_count += 1
        nb_confusion_matrix[actual_orientation][predicted_orientation] += 1
        nb_confusion_matrix[actual_orientation]["Actual Count"] += 1

    # Prints the confusion matrix
    print_confusion_matrix(nb_confusion_matrix)
    # Calculates the accuracy
    nnet_accuracy = calc_accuracy(test_correct_count,len(test_grid))
    return test_output_dict, nnet_accuracy


###########################################End of Neural Net Code##############################################



# Called to initialize the program
def main(train_file, test_file, algorithm): #, stump_count):#, test_file, algorithm, stump_count):
    orientation_list = get_orientation_list()
    if algorithm == 'adaboost':
        print "Performing AdaBoost Algorithm..."
        print "Best stump_count: 100\nExpected Running Time: < 15-20 seconds"
        print "Thank you for waiting...\nWe appreciate your patience..."
        stump_count = int(sys.argv[4])
        train_file_dict = get_train_dict(train_file, algorithm)
        test_file_dict = get_test_dict(test_file, algorithm)
        trained_model_dict = train_adaboost(train_file_dict, stump_count, orientation_list)
        test_output_dict, adaboost_accuracy = test_adaboost(trained_model_dict, test_file_dict, stump_count, orientation_list)
        write_test_output_file(test_output_dict, algorithm + "_output.txt")
        print "Adaboost Accuracy: ", adaboost_accuracy
        print "Testing on AdaBoost Complete..."
        print "Thank you for your time and patience...have a great day :).."
    elif algorithm == 'nearest' or algorithm == 'best':
        print "Performing k-nearest neighbours algorithm..."
        print "Best k: 50\nExpected Running Time: ~8-9 mins"
        print "This may take a while...thank you for waiting...\nWe appreciate your patience..."
        k_value = 50
        train_fname_label_list, train_grid = get_train_dict(train_file, algorithm)
        test_fname_label_list, test_grid = get_test_dict(test_file, algorithm)
        test_output_dict, kNN_accuracy = test_kNN(train_fname_label_list, train_grid, test_fname_label_list, test_grid, k_value, orientation_list)
        write_test_output_file(test_output_dict, algorithm + "_output.txt")
        print "K = ", k_value
        print "kNN Accuracy: ", kNN_accuracy
        print "Testing on k-nearest neighbours complete..."
        print "Thank you for your time and patience...have a great day :).."
    elif algorithm == 'nnet':
        print "Performing Neural Net Algorithm..."
        print "Best hidden_count: 25\nExpected Running Time: ~7-8 mins"
        print "This may take a while...thank you for waiting...\nWe appreciate your patience..."
        print "You may try with smaller hidden count (hc = 10) for better performance..."
        # hidden_count = stump_count
        hidden_count = int(sys.argv[4])
        # train_fname_label_list, train_grid = get_train_dict(train_file, algorithm)
        train_fname_label_list, train_label, train_grid = get_train_dict(train_file, algorithm)
        test_fname_label_list, test_label, test_grid = get_test_dict(test_file, algorithm)
        nnet_model = defaultdict(int)
        nnet_model = train_nnet(train_fname_label_list, train_label, train_grid, hidden_count, orientation_list)
        write_data_to_file("nnet_model.p", nnet_model)
        test_output_dict, nnet_accuracy = test_nnet(nnet_model, test_fname_label_list, test_label, test_grid, hidden_count, orientation_list)
        write_test_output_file(test_output_dict, algorithm + "_output.txt")
        print "Hidden Layer Node Count = ", hidden_count
        print "Neural Net Accuracy: ", nnet_accuracy
        print "Testing on Neural Net Complete..."
        print "Thank you for your time and patience...have a great day :).."

if __name__ == '__main__':
    start_time = time.time()
    train_file = str(sys.argv[1])
    test_file = str(sys.argv[2])
    algorithm = str(sys.argv[3])
    main(train_file, test_file, algorithm)

    # Adaboost
    # main("train-data.txt", "test-data.txt", "adaboost", 100)
    # kNN
    # main("train-data.txt", "test-data.txt", "nearest", 50)
    # nnet
    # main("train-data.txt", "test-data.txt", "nnet", 25)
    end_time = time.time()
    print "Total Time", end_time - start_time
