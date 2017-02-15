# Author: Rahul Sampat
# Nov 2016

# NOTE: THIS PROGRAM IS COMPATIBLE WITH JUST 3.4 OR HIGHER VERSION!!!
# KINDLY RUN IT ON A COMPATIBLE PYTHON VERSION
# MAY GIVE ERRORS/INCORRECT RESULTS ON PYTHON 2.7

# This is the main file to run the program.
# Also there are 3 other files (used for decision tree).
# Please keep them in the same folder as well.
# The other 3 files are as follows:
# 1) functions_dt.py
# 2) main_dt.py
# 3) Node.py

# *********************************************Naive Bayes**********************************************************#
# Part 1a - Naive Bayes Classifier
# Formulation of the problem:
# Part 1a of the problem was done by implementing a Naive Bayes Classifier
# Data Pre-processing - Word Sanitization - I tried sanitizing the text in various ways like stripping,
# converting to lower, removing punctuations and stops words,only to later on realize that the best accuracy
# for Naive Bayes was achieved by not doing anything at all.

# I am using pickle package of python to store files in filename.p format.
# The object/variable can directly be retrieved from this file and can be stored in a variable directly.

# Note: Please run the code to get the Accuracy Results, Confusion Matrix and the Top & Least Spam Words

# Working:
# Does the below procedure for both Binary and Continuous Classifier
# 1) Training the classifiers:
# a) Creates Bag of words for both spam and non-spam documents and stores in a dict.
# b) Calculates prior probabilities & likelihood for them
# c) Stores the data in the pickle file
# d) Print the Top & Least 10 words for Binary and Continuous
#
# I am calculating the Top & Least Spam words with the below formula:
# P(S=1|w_i) = P(w_i|S=1)/P(w_i|S=1) + P(w_i/S=0)
# I am ignoring P(S=1), since its a constant for each topic and we have to find the top words per topic and
# not across topics.

# 2) Testing the classifier:
# a) Reads the trained data from the pickle file and stores in the respective variables.
# b) Runs the learned Decision Tree on the test data
# c) Creates and prints the confusion matrix
# d) Creates and prints the accuracy

# Note: Please run the code to get the Accuracy Results, Confusion Matrix and the Top & Least Spam Words

# Naive Bayes Results:
# Binary Naive Bayes Top 10 Spam Associated Words:
# 1. <jm@netnoteinc.com>;
# 2. zzzzason.org
# 3. ([213.105.180.140])
# 4. zzzz@localhost.spamassassin.taint.org
# 5. <zzzz@jmason.org>;
# 6. mail.webnote.net
# 7. [193.120.211.219]
# 8. yyyy@netnoteinc.com
# 9. zzzz@localhost.jmason.org
# 10. <webmaster@efi.ie>;
#
# Binary Naive Bayes Least 10 Spam Associated Words:
# 1. X-Spam-Status:
# 2. version=2.50-cvs
# 3. yyyy@localhost.example.com
# 4. required=5.0
# 5. In-Reply-To:
# 6. required=7.0
# 7. References:
# 8. fork@example.com
# 9. <mailto:fork@example.com>
# 10. yyyy@example.com
#
#
# Continuous Naive Bayes Top 10 Spam Associated Words:
# 1. mv
# 2. face="Verdana"><font
# 3. <blockquote><font
# 4. hq.pro-ns.net
# 5. <jm@netnoteinc.com>;
# 6. type=3D"text"
# 7. width=3D"15%"
# 8. width=3D"550"
# 9. 0px;
# 10. face=3D"Verdana,
#
# Continuous Naive Bayes Least 10 Spam Associated Words:
# 1. ////////////////////////////////////////////////////////////////////////////
# 2. X-Spam-Level:
# 3. X-Spam-Status:
# 4. fork@example.com
# 5. src="http://www.cnet.com/b.gif"
# 6. version=2.50-cvs
# 7. yyyy@localhost.example.com
# 8. src="http://www.zdnet.com/b.gif"
# 9. src="http://home.cnet.com/b.gif"
# 10. <rssfeeds@example.com>
#
#
# Binary Naive Bayes Confusion Matrix
# 	 Model Results
#   	 T 	    N 	Total
# T 	1170    15	1185
# N 	26    1343	1369
#
#
#
# Continuous Naive Bayes Confusion Matrix
# 	 Model Results
#   	 T 	    N 	Total
# T 	1141    44	1185
# N 	22    1347	1369
#
#
# Naive Bayes Binary Accuracy 98.39467501957714
# Naive Bayes Continuous Accuracy 97.41581832419733
# Naive Bayes Average Accuracy 97.90524667188723

# As we can see, the Binary Features for Naive Bayes works slightly better than with Continuous Features
# Also, word sanitization hampers the performance/accuracy of the classifier

# *********************************************Decision Tree**********************************************************#
# Part1b - Decision Tree
# Formulation of the problem:
# Part 1a of the problem was done by implementing a Naive Bayes Classifier
# Data Pre-processing - Word Sanitization - For DT, I did sanitize the words by stripping
# and converting them to lower case.
# It actually improved the accuracy

# Working:
# Does the below procedure for both Binary and Continuous Classifier
# 1) Training the classifiers:
# a) Creates Bag of words (and simultaneously sanitizes them as well )for both spam and non-spam documents
# and stores in a dict
# b) Filters the top words (whose frequency is more than a certain threshold-passed as a
# parameter to update_all_words_list() function). For now, I am considering the words whose frequency is more than 300.
# c) Stores the class labels and above selected most frequent words as their features as a training data set
# into a csv file (for for Binary and Continuous features separately),which are later used to build the tree.
# The csv format - Saves Emails as rows and Words as feature columns. The first column is the class label.
# d) Writes all the variables in a pickle file
# e) Builds and Prints the Decision Tree for both Binary and Continuous Features
#
# 2) Testing the classifier:
# a) Reads all the variables from a pickle file
# b) Reads the test data and creates a similar csv file as created for training data above
# c) Runs the Decision Tree Classifier to predict the label of the test
# d) Calculates and prints the accuracy and prints the confusion matrix for both type of features

# Design Decisions:
# I am converting the data to a csv and then using it to build the tree

# I have used Pandas for storing the data in the 'Dataframe' data structure and some of its basic
# functions of it to find row/column count and filtering rows/column.
# We haven't used any advanced functionality of it.

# We are calculating Information Gain = (Entropy before split-Entropy after split) and correspondingly
# choosing the best attribute and its respective value

# Also for continuous values, if the distinct values is more than 4, then we divide the whole list into 2 lists
# and take the medium
# of the respective sub lists for better splitting and efficiency

# Results for Decision Tree Algorithm:
# Tree for BINARY Data Set till the depth of 5:
# Node
# column: x-spam-status:  value: 0  depth: 0  label: None  ld: 1535  rd: 1111
# left child
# Node
# column: in-reply-to:  value: 0  depth: 1  label: None  ld: 1420  rd: 115
# left child
# Node
# column: |  value: 0  depth: 2  label: None  ld: 1355  rd: 65
# left child
# Node
# column: precedence:  value: 0  depth: 3  label: None  ld: 1064  rd: 291
# left child
# Node
# column: id  value: 0  depth: 4  label: None  ld: 11  rd: 1053
# left child
# Leaf node
# column: return-path:  value: 0  depth: 5  label: 0
# right child
# Leaf node
# column: &nbsp;<a  value: 0  depth: 5  label: 1
# right child
# Node
# column: sender:  value: 0  depth: 4  label: None  ld: 36  rd: 255
# left child
# Leaf node
# column: jun  value: 0  depth: 5  label: 0
# right child
# Leaf node
# column: jul  value: 0  depth: 5  label: 1
# right child
# Node
# column: color:  value: 0  depth: 3  label: None  ld: 60  rd: 5
# left child
# Node
# column: 26  value: 0  depth: 4  label: None  ld: 58  rd: 2
# left child
# Leaf node
# column: <tr><td  value: 0  depth: 5  label: 0
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 5  label: 1
# right child
# Node
# column: think  value: 0  depth: 4  label: None  ld: 4  rd: 1
# left child
# Leaf node
# column: <tr><td  value: 0  depth: 5  label: 1
# right child
# Leaf node
# column: <tr><td  value: 1  depth: 5  label: 0
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 2  label: 0
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 1  label: 0
#
#
# Tree for CONTINUOUS Data Set till the depth of 5:
# Node
# column: x-spam-status:  value: 0  depth: 0  label: None  ld: 1535  rd: 1111
# left child
# Node
# column: in-reply-to:  value: 0  depth: 1  label: None  ld: 1420  rd: 115
# left child
# Node
# column: <!--  value: 8  depth: 2  label: None  ld: 1357  rd: 63
# left child
# Node
# column: precedence:  value: 0  depth: 3  label: None  ld: 1065  rd: 292
# left child
# Node
# column: [127.0.0.1])  value: 1  depth: 4  label: None  ld: 1051  rd: 14
# left child
# Leaf node
# column: >>  value: 0  depth: 5  label: 1
# right child
# Leaf node
# column: jm@localhost  value: 0  depth: 5  label: 0
# right child
# Node
# column: sender:  value: 0  depth: 4  label: None  ld: 39  rd: 253
# left child
# Leaf node
# column: jun  value: 0  depth: 5  label: 0
# right child
# Leaf node
# column: <zzzz@localhost>;  value: 0  depth: 5  label: 1
# right child
# Node
# column: build  value: 1  depth: 3  label: None  ld: 59  rd: 4
# left child
# Leaf node
# column: <tr><td  value: 7  depth: 4  label: 0
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 4  label: 1
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 2  label: 0
# right child
# Leaf node
# column: <tr><td  value: 0  depth: 1  label: 0
#
# Binary Results:
# Binary Accuracy: 95.69303054032889
# Confusion Matrix:
#
# Binary Confusion Matrix
# 	 Model Results
#   	 T 	    N 	Total
# T 	1177    102	1185
# N 	8    1267	1369
#
#
# Continuous Results:
# Continuous Accuracy: 96.39780736100235
# Confusion Matrix:
#
# Continuous Confusion Matrix
# 	 Model Results
#   	 T 	    N 	Total
# T 	1178    85	1185
# N 	7    1284	1369

# As we can see, the Continuous Features for Decision Tree works slightly better than with Binary Features
# Also word sanitization - stripping, converting to lower, removing punctuations and stops words. works in favor of
# Decision Tree Classifier and improves its accuracy, which is not the case with Naive Bayes Classifier.


# Naive Bayes VS Decision Tree for Spam Classification:
# I would say that Naive Bayes works better for the Spam Classification Task over Decision Tree algorithm
# for the following reasons:
# 1) Greater Accuracy (There is a slight difference, but it may depend on the data set as well)
# 2) Naive Bayes performs much faster as compared to Decision Tree:
# (NB- ~20-30 secs) vs (DT-300-400 secs)
# which makes Decision Tree almost 10 times slower, and the difference would exponentially increase with the size of
# the data set
# 3) If all the words/features from the training data set are taken into consideration for the Decision Tree, then it
# will become even more slower and might also over fit the data. We do not face such issues with Naive Bayes



import sys
import os
from collections import defaultdict
import string
import math as m
import time
import collections
import numpy as np
import pandas as ps
import copy
import pickle
from main_dt import decision_tree_test, decision_tree_train

# ************************************************Start of Naive Bayes Code*******************************************#

# Global Variables
file_list = []

nb_all_words = defaultdict(int)
nb_bag_of_words_dict = defaultdict(int)

nb_file_count_dict = defaultdict(int)
nb_word_count_dict = defaultdict(int)

nb_binary_prob_dict = defaultdict(int)
nb_cont_prob_dict = defaultdict(int)

subdir_list = ['spam', 'notspam']

# directory = "E:\\Rahul\\IUB Education\\Fall 2016\\AI (David Crandell)\\Assignments\\Assignment-4\\rrsampat-vsureshb-anveling-a4\\part1"
# directory = "E:\\Rahul\\IUB Education\\Fall 2016\\AI (David Crandell)\\Assignments\\Assignment-4\\part1"


# Sanitizes the word for NB
def nb_sanitize_word(word):
    # word = word.strip()
    # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # word = word.translate(replace_punctuation)
    # word = word.lower()
    # word = word.translate(None, string.punctuation)
    return word


# Creates confusion matrix dictionary
def create_confusion_matrix():
    confusion_matrix = defaultdict(int)
    # confusion_list = ['TP', 'FP', 'TN', 'FN', 'AP', 'AN']
    # for key in confusion_list:
    #     confusion_matrix[key] = 0
    # confusion_matrix = {'TP': 0, 'FP':0, 'TN':0, 'FN':0, 'AP':0, 'AN':0}
    confusion_matrix['TP'] = 0
    confusion_matrix['FP'] = 0
    confusion_matrix['FN'] = 0
    confusion_matrix['TN'] = 0
    confusion_matrix['AP'] = 0
    confusion_matrix['AN'] = 0
    return confusion_matrix


# Writes all the dicts to the text files
def write_all_dict_to_files(file_name, technique):
    # write_data_to_file("nb_all_words.p", nb_all_words)
    # write_data_to_file("nb_bag_of_words_dict.p", nb_bag_of_words_dict)
    # write_data_to_file("nb_file_count_dict.p", nb_file_count_dict)
    # write_data_to_file("nb_word_count_dict.p", nb_word_count_dict)
    # write_data_to_file("nb_binary_prob_dict.p", nb_binary_prob_dict)
    # write_data_to_file("nb_cont_prob_dict.p", nb_cont_prob_dict)
    write_data_to_file(file_name, technique)


# Writes a particular dict to the text files
def write_data_to_file(file_name, technique):
    print("Writing files to pickle...")
    global file_list
    # [nb_file_count_dict, nb_word_count_dict, nb_binary_prob_dict, nb_cont_prob_dict]
    if technique == 'bayes':
        file_list.append(nb_file_count_dict)
        file_list.append(nb_word_count_dict)
        file_list.append(nb_binary_prob_dict)
        file_list.append(nb_cont_prob_dict)
    elif technique == 'dt':
        file_list.append(dt_all_words_dict)
        file_list.append(dt_all_words_list)
        file_list.append(dt_train_binary_bag_of_words)
        file_list.append(dt_train_cont_bag_of_words)

    pickle.dump(file_list, open(file_name, "wb"))
    # pickle.dump( dictname, open( filename, "wb" ) )
    # with open(filename, 'w') as f:
    #     pickle.dump([nb_file_count_dict, nb_word_count_dict, nb_binary_prob_dict, nb_cont_prob_dict], f)


# Reads all the dicts to the text files
def reads_all_dict_to_files(file_name, technique):
    # nb_all_words = read_data_from_file("nb_all_words.p")
    # nb_bag_of_words_dict = read_data_from_file("nb_bag_of_words_dict.p")
    # nb_file_count_dict = read_data_from_file("nb_file_count_dict.p")
    # nb_word_count_dict = read_data_from_file("nb_word_count_dict.p")
    # nb_binary_prob_dict = read_data_from_file("nb_binary_prob_dict.p")
    # nb_cont_prob_dict = read_data_from_file("nb_cont_prob_dict.p")
    read_data_from_file(file_name, technique)


# Reads the dict from the text files and stores in the dict
def read_data_from_file(file_name, technique):
    print("Reading files from pickle...")
    global nb_all_words, nb_bag_of_words_dict, nb_file_count_dict, nb_word_count_dict, nb_binary_prob_dict, \
        nb_cont_prob_dict, file_list, dt_all_words_dict, dt_all_words_list, dt_train_binary_bag_of_words, dt_train_cont_bag_of_words
    file_list = pickle.load(open(file_name, "rb"))
    if technique == 'bayes':
        nb_file_count_dict = file_list[0]
        nb_word_count_dict = file_list[1]
        nb_binary_prob_dict = file_list[2]
        nb_cont_prob_dict = file_list[3]
    if technique == 'dt':
        dt_all_words_dict = file_list[0]
        dt_all_words_list = file_list[1]
        dt_train_binary_bag_of_words = file_list[2]
        dt_train_cont_bag_of_words = file_list[3]

        # filename = pickle.load( open( filename, "rb" ))
        # return filename
        # with open(filename) as f:
        #     nb_file_count_dict, nb_word_count_dict, nb_binary_prob_dict, nb_cont_prob_dict = pickle.load(f)


# Returns a list of words in a given email file
def create_email_string(file_path):
    try:
        f = open(file_path, encoding="Latin-1")
        email_string = f.read().split()
        f.close()
        return email_string
    except IOError:
        print("Cannot read from file:", file_path)
        return 0


# Finds and prints the top & least 10 spam associated words
def print_spam_words(prob_dict, nb_type):
    # train_prob_spam = (1.0 * nb_file_count_dict["spam"]) / (nb_file_count_dict["spam"] + nb_file_count_dict["notspam"])
    spam_word_dict = defaultdict(int)
    if nb_type == "Binary":
        total_spam = len(prob_dict["spam"])
        total_notspam = len(prob_dict["notspam"])
    else:
        total_spam = nb_word_count_dict["spam"]
        total_notspam = nb_word_count_dict["notspam"]

    for word in nb_all_words:
        denominator = prob_dict["spam"][word] if prob_dict["spam"][word] != 0 else (1.0 / total_spam)
        denominator += prob_dict["notspam"][word] if prob_dict["notspam"][word] != 0 else (1.0 / total_notspam)
        numerator = prob_dict["spam"][word] if prob_dict["spam"][word] != 0 else (1.0 / total_spam)
        word_spam_prob = m.log(numerator) - m.log(denominator)
        spam_word_dict[word] = word_spam_prob

    spam_word_dict_ordered = collections.OrderedDict()
    for k in sorted(spam_word_dict, key=lambda k: spam_word_dict[k], reverse=True):
        spam_word_dict_ordered[k] = spam_word_dict[k]
    # print "Spam Word Ordered Dict", spam_word_dict_ordered
    print(nb_type + " Top 10 Spam Associated Words:")
    for x in range(10):
        print(str(x + 1) + ". " + list(spam_word_dict_ordered.keys())[x])
    print("\n" + nb_type + " Least 10 Spam Associated Words:")
    n = 1
    for y in range(len(spam_word_dict_ordered) - 1, len(spam_word_dict_ordered) - 11, -1):
        print(str(n) + ". " + list(spam_word_dict_ordered.keys())[y])
        n += 1
    print("\n")


# Reads the training data from the given directory
# dataset = "train"
# subdir = "spam"/"notspam"
def nb_train_data(dir_path, dataset, filename):
    print("Training Naive Bayes Classifier...")
    print("Thank you for waiting...\nWe appreciate your patience...")
    directory_path = dir_path + os.path.sep  # + dataset
    # for dir in os.walk(directory_path):
    #   for subdir in dir[1]:
    for subdir in os.listdir(directory_path):
        if subdir in subdir_list:
            subdir_path = directory_path + os.sep + subdir + os.sep
            nb_create_bag_of_words(subdir_path, subdir)
            nb_create_prior_prob_dicts(subdir)
    print_spam_words(nb_binary_prob_dict, "Binary Naive Bayes")
    print_spam_words(nb_cont_prob_dict, "Continuous Naive Bayes")


# Reads & creates the Spam word dictionary. Also updates the nb_all_words dictionary
# subdir = "spam"/"notspam"
def nb_create_bag_of_words(directory_path, subdir):
    global nb_all_words, nb_bag_of_words_dict, nb_file_count_dict, nb_word_count_dict
    file_count = 0
    word_count = 0
    if nb_bag_of_words_dict.get(subdir, 0) == 0:
        nb_bag_of_words_dict[subdir] = defaultdict(int)
    for email in os.listdir(directory_path):
        file_count += 1
        file_path = directory_path + email
        email_string = create_email_string(file_path)
        for word in email_string:
            word = nb_sanitize_word(word)
            word_count += 1
            nb_all_words[word] += 1
            if nb_bag_of_words_dict[subdir].get(word, 0) == 0:
                nb_bag_of_words_dict[subdir][word] = defaultdict(int)
                nb_bag_of_words_dict[subdir][word][file_count] += 1
            else:
                nb_bag_of_words_dict[subdir][word][file_count] += 1
    nb_file_count_dict[subdir] = file_count
    nb_word_count_dict[subdir] = word_count


# Creates the Binary & Continuous probabilities for all the words in the Spam & Not Spam word dictionary.
# subdir = "spam"/"notspam"
def nb_create_prior_prob_dicts(subdir):
    global nb_binary_prob_dict, nb_cont_prob_dict
    nb_binary_prob_dict[subdir] = defaultdict(int)
    nb_cont_prob_dict[subdir] = defaultdict(int)
    for word in nb_bag_of_words_dict[subdir]:
        nb_binary_prob_dict[subdir][word] = (1.0 * len(nb_bag_of_words_dict[subdir][word])) / nb_file_count_dict[subdir]
        nb_cont_prob_dict[subdir][word] = (1.0 * sum(nb_bag_of_words_dict[subdir][word].values())) / nb_word_count_dict[
            subdir]


# Calculates accuracy based on the given confusion matrix
def calc_accuracy(confusion_matrix):
    return (100.0 * (confusion_matrix['TP'] + confusion_matrix['TN'])) / (confusion_matrix['AP'] + confusion_matrix['AN'])


# Print confusion matrix
def print_confusion_matrix(confusion_matrix, nb_type):
    print(nb_type + " Confusion Matrix")
    print("\t Model Results")
    print("  \t T \t    N \tTotal")
    print("T \t" + str(confusion_matrix['TP']) + "    " + str(confusion_matrix['FN']) + "\t" + str(
        confusion_matrix['AP']))
    print("N \t" + str(confusion_matrix['FP']) + "    " + str(confusion_matrix['TN']) + "\t" + str(
        confusion_matrix['AN']))
    print("\n")


# Reads the test data from the given directory
# dataset = "test"
# subdir = "spam"/"notspam"
def nb_test_data(dir_path, dataset, prob_dict, nb_type, file_name):
    # test_file_count = 0
    # test_correct_count = 0
    train_prob_spam = (1.0 * nb_file_count_dict["spam"]) / (nb_file_count_dict["spam"] + nb_file_count_dict["notspam"])
    train_prob_not_spam = (1.0 * nb_file_count_dict["notspam"]) / (
        nb_file_count_dict["spam"] + nb_file_count_dict["notspam"])
    nb_confusion_matrix = create_confusion_matrix()

    directory_path = dir_path + os.path.sep  # + dataset
    # for dir in os.walk(directory_path):
    #     for subdir in dir[1]:
    for subdir in os.listdir(directory_path):
        if subdir in subdir_list:
            subdir_path = directory_path + os.sep + subdir + os.sep
            for email in os.listdir(subdir_path):
                # test_file_count += 1
                # nb_confusion_matrix = naive_bayes(subdir_path + email, train_prob_spam, train_prob_not_spam, subdir, prob_dict, nb_confusion_matrix)
                # test_correct_count += naive_bayes(subdir_path + email, train_prob_spam, train_prob_not_spam, subdir, prob_dict, nb_confusion_matrix)
                naive_bayes(subdir_path + email, train_prob_spam, train_prob_not_spam, subdir, prob_dict,
                            nb_confusion_matrix)

    print_confusion_matrix(nb_confusion_matrix, nb_type)
    return calc_accuracy(nb_confusion_matrix)
    # return (100.0 * test_correct_count)/test_file_count


# Calculates the posterior probabilities P(S|all words in the email) for all the files in filepath
def naive_bayes(file_path, train_prob_spam, train_prob_not_spam, label, prob_dict, nb_confusion_matrix):
    prob_spam = m.log(train_prob_spam)
    prob_not_spam = m.log(train_prob_not_spam)
    email_string = create_email_string(file_path)
    for word in email_string:
        word = nb_sanitize_word(word)
        prob_spam += m.log(prob_dict["spam"][word]) if prob_dict["spam"][word] != 0 else m.log(
            0.5 / len(prob_dict["spam"]))
        prob_not_spam += m.log(prob_dict["notspam"][word]) if prob_dict["notspam"][word] != 0 else m.log(
            0.5 / len(prob_dict["notspam"]))

    if (prob_spam > prob_not_spam and label == "spam"):
        nb_confusion_matrix['TP'] += 1
        nb_confusion_matrix['AP'] += 1
        return 1
    elif (prob_not_spam > prob_spam and label == "notspam"):
        nb_confusion_matrix['TN'] += 1
        nb_confusion_matrix['AN'] += 1
        return 1
    elif (prob_spam > prob_not_spam and label == "notspam"):
        nb_confusion_matrix['FP'] += 1
        nb_confusion_matrix['AN'] += 1
        return 0
    elif (prob_not_spam > prob_spam and label == "spam"):
        nb_confusion_matrix['FN'] += 1
        nb_confusion_matrix['AP'] += 1
        return 0
    return nb_confusion_matrix


# ***********************************************************End of Naive Bayes Code********************************************** #

# ***********************************************************Start of Decision Tree Code****************************************** #
# Global Variables
dt_all_words_dict = defaultdict(int)
dt_all_words_list = []
dt_train_binary_bag_of_words = []
dt_train_cont_bag_of_words = []
dt_test_binary_bag_of_words = []
dt_test_cont_bag_of_words = []


# Sanitizes the word for NB
def dt_sanitize_word(word):
    word = word.strip()
    # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # word = word.translate(replace_punctuation)
    word = word.lower()
    # word = word.translate(None, string.punctuation)
    return word


# Creates a stop words dict
def get_stop_words_dict():
    stop_words_list = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
                       "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
                       "amongst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway",
                       "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes",
                       "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                       "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant",
                       "co", "con", "could", "couldn't", "cry", "de", "describe", "detail", "do", "done", "down", "due",
                       "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
                       "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few",
                       "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty",
                       "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasn't",
                       "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
                       "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc",
                       "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly",
                       "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine",
                       "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
                       "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "no one", "nor",
                       "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
                       "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part",
                       "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
                       "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six",
                       "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
                       "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
                       "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these",
                       "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout",
                       "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
                       "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
                       "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
                       "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",
                       "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
                       "yours", "yourself", "yourselves", "the"]
    # stop_words_list = ["a"]
    stop_words_dict = defaultdict(int)
    for word in stop_words_list:
        stop_words_dict[word] = 1
    return stop_words_dict


# Stores the bag of words for the dataset in a pandas csv
def create_pandas_csv(binary_bog_list, cont_bog_list, col_headers, dataset):
    pan_binary = ps.DataFrame(binary_bog_list, columns=col_headers)
    pan_cont = ps.DataFrame(cont_bog_list, columns=col_headers)
    pan_binary.to_csv('panBinary_' + dataset + '.csv', index=False)
    pan_cont.to_csv('panCont_' + dataset + '.csv', index=False)


# Reads the training data from the given directory
# dataset = "train"
# subdir = "spam"/"notspam"
def dt_train_data(dir_path, dataset):
    print("Training Decision Tree Classifier")
    print("Thank you for waiting...\nWe appreciate your patience...")
    global dt_train_binary_bag_of_words, dt_train_cont_bag_of_words
    directory_path = dir_path + os.path.sep  # + dataset
    # Gets all the words from the spam and non-spam email and stores in a dict along with their frequencies
    dt_update_all_words_dict(directory_path)
    # Gets the words from the above dict whose frequency is more than passed frequency parameter
    update_all_words_list(300)
    for subdir in os.listdir(directory_path):
        if subdir in subdir_list:
            label = 1 if subdir == 'spam' else 0  #######################Change to 'spam'
            # label = label if data set == "train" else -1
            subdir_path = directory_path + os.sep + subdir + os.sep
            for email in os.listdir(subdir_path):
                file_path = subdir_path + email
                # Creates a bag of words dataset with email as rows and words as feature columns. The first column is the class label.
                binary_email_list, cont_email_list = create_bow_lists(file_path, label)
                dt_train_binary_bag_of_words.append(binary_email_list)
                dt_train_cont_bag_of_words.append(cont_email_list)
    # Using this list to update the column headers
    # column_headers = copy.deepcopy(dt_all_words_list)
    column_headers = ['laalbel'] + dt_all_words_list
    # print("Length local all words", len(column_headers))
    # print("Binary Word List", len(dt_train_binary_bag_of_words))
    # print("Cont Word List", len(dt_train_cont_bag_of_words))
    # Stores the train bag of words for the dataset in a csv
    create_pandas_csv(dt_train_binary_bag_of_words, dt_train_cont_bag_of_words, column_headers, dataset)
    # Making decision tree
    decision_tree_train("train")


# Reads the training data from the given directory
# dataset = "train"
# subdir = "spam"/"notspam"
def dt_test_data(dir_path, dataset):
    global dt_test_binary_bag_of_words, dt_test_cont_bag_of_words
    directory_path = dir_path + os.path.sep  # + dataset
    # No need to call this for testing
    # Gets all the words from the spam and non-spam email and stores in a dict along with their frequencies
    # dt_update_all_words_dict(directory_path)
    # Gets the words from the above dict whose frequency is more than passed frequency parameter
    # No need to call this for testing
    # update_all_words_list(180)
    for subdir in os.listdir(directory_path):
        if subdir in subdir_list:
            label = 1 if subdir == 'spam' else 0  #######################Change to 'spam'
            subdir_path = directory_path + os.sep + subdir + os.sep
            for email in os.listdir(subdir_path):
                file_path = subdir_path + email
                # Creates a bag of words dataset with email as rows and words as feature columns. The first column is the class label.
                binary_email_list, cont_email_list = create_bow_lists(file_path, label)
                dt_test_binary_bag_of_words.append(binary_email_list)
                dt_test_cont_bag_of_words.append(cont_email_list)
    # Using this list to update the column headers
    # column_headers = copy.deepcopy(dt_all_words_list)
    column_headers = ['laalbel'] + dt_all_words_list
    # print("Length local all words", len(column_headers))
    # print("Binary Word List", len(dt_test_binary_bag_of_words))
    # print("Cont Word List", len(dt_test_cont_bag_of_words))
    # Stores the test bag of words for the data set in a csv
    create_pandas_csv(dt_test_binary_bag_of_words, dt_test_cont_bag_of_words, column_headers, dataset)
    decision_tree_train(mode)


# Updates all words global list, but skips the words whose frequency is below word frequency parameter
def update_all_words_list(frequency_threshold):
    global dt_all_words_list
    for word, frequency in dt_all_words_dict.items():
        if frequency > frequency_threshold:
            dt_all_words_list.append(word)
    # print("Total Words:", len(dt_all_words_list))


# Reads & creates the dictionary. Also updates the nb_all_words dictionary
# label =  1 / 0
def dt_update_all_words_dict(directory_path):
    global dt_all_words_dict
    stop_words = get_stop_words_dict()
    for subdir in os.listdir(directory_path):
        if subdir in subdir_list:
            subdir_path = directory_path + os.sep + subdir + os.sep
            for email in os.listdir(subdir_path):
                file_path = subdir_path + email
                email_string = create_email_string(file_path)
                for word in email_string:
                    word = dt_sanitize_word(word)
                    if word not in stop_words:
                        dt_all_words_dict[word] += 1


# Creates a bag of words dataset with email as rows and words as feature columns. The first column is the class label.
def create_bow_lists(file_path, label):
    binary_email_list = [0 for x in range(len(dt_all_words_list))]
    cont_email_list = [0 for x in range(len(dt_all_words_list))]
    email_string = create_email_string(file_path)
    for word in email_string:
        word = dt_sanitize_word(word)
        if word in dt_all_words_list:
            binary_email_list[dt_all_words_list.index(word)] = 1
            cont_email_list[dt_all_words_list.index(word)] += 1
    binary_email_list = [label] + binary_email_list
    cont_email_list = [label] + cont_email_list
    return binary_email_list, cont_email_list
    # print "Binary Email List Length:", len(binary_email_list)
    # print "Cont Email List Length:", len(cont_email_list)


# ***********************************************************End of Decision Tree Code****************************************** #
# Called to initialize the program
def main(mode, technique, directory, file_name):
    # technique = 'bayes'
    if technique == 'bayes':
        if mode == "train":
            nb_train_data(directory, mode, file_name)
            write_all_dict_to_files(file_name, technique)
            print("Training Naive Bayes Complete")
            print("Thank you for your time and patience...have a great day :)..")
        elif mode == "test":
            print("Testing Naive Bayes Classifier...")
            print("Thank you for waiting...\nWe appreciate your patience...")
            reads_all_dict_to_files(file_name, technique)
            nb_binary_accuracy = nb_test_data(directory, mode, nb_binary_prob_dict, "Binary Naive Bayes", file_name)
            nb_continuous_accuracy = nb_test_data(directory, mode, nb_cont_prob_dict, "Continuous Naive Bayes",
                                                  file_name)
            print("Naive Bayes Binary Accuracy", nb_binary_accuracy)
            print("Naive Bayes Continuous Accuracy", nb_continuous_accuracy)
            print("Naive Bayes Average Accuracy", float(nb_binary_accuracy + nb_continuous_accuracy) / 2.0)
            print("Testing Naive Bayes Complete")
            print("Thank you for your time and patience...have a great day :)..")
    elif technique == 'dt':
        if mode == "train":
            dt_train_data(directory, mode)
            write_all_dict_to_files(file_name, technique)
            print("Training Decision Tree Complete")
            print("Thank you for your time and patience...have a great day :)..")
        elif mode == "test":
            print("Testing Decision Tree Classifier")
            print("This may take a while...thank you for waiting...\nWe appreciate your patience...")
            # Creates csv files for the test data
            reads_all_dict_to_files(file_name, technique)
            dt_test_data(directory, mode)
            print("Testing Decision Tree Complete")
            print("Thank you for your time and patience...have a great day :)..")


if __name__ == '__main__':
    start_time = time.time()
    mode = str(sys.argv[1])
    technique = str(sys.argv[2])
    directory = str(sys.argv[3])
    file_name = str(sys.argv[4])
    file_name += ".p"
    main(mode, technique, directory, file_name)
    end_time = time.time()
    print("Total Time", end_time - start_time)
