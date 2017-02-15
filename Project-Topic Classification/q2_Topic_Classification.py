
# Q2 - Topic Classification
# In this problem, we had to classify the documents among the given 20 topics, the catch being we can observe
# the labels of only a fraction of the training set.

# NOTE: THIS CODE IS PYTHON 2.7 COMPATIBLE. PLEASE DO NOT RUN IT ON PYTHON 3 VERSION!!


# Due to a huge data set, the code takes 5+ mins to execute completely.

# We have used the Naive Bayes algorithm along with Expectation Maximization to first predict the topics/labels
# of the unknown documents from the training data set itself.
# We re-iterate the Expectation Maximization to a point where the probabilities converge & label the unlabelled data set
# Once the training data set is completed labelled, we calculate the priors to label the test data set.

# We are doing that as follows:
# 1) We have first divided the data set into 2 parts-Labelled and Unlabelled
# 2) In Expectation Maximization, we are then calculating the priors and the likelihood using just the labelled
# trained data set
# 3) Now using these priors, we are calculating the probability of the labels of the documents in unlabelled
# training data set
# 4) Now for a document in an unlabelled data set, we get some probability of it being in each of the topics.
# 5) Now taking these probabilities of the unlabelled data set, along with the previous probabilities of the
# labelled dataset,
# we are recalculating the priors.
# 6) We are doing this for 5 iterations, and expecting the probabilities of the labels to converge by then.
# 7) After 5 iterations, we are taking the maximum probability of each document being in a class/topic and
# assigning the corresponding topic to that document.
# 8) Once the training data set is completed labelled (each document belonging to just one topic
# with a probability of 1), we re-calculate the priors one last time.
# 9) We are then using these priors and likelihood to predict the labels of the test data set.

# Design Decisions & Challenges:
# 1) It was really difficult to come with a program that would handle such a large data set.
# I tried creating a NxN numpy array for storing the table with the information of word vs document vs topic presence,but the matrix
# was too large and hence was really slow at performing the task.
# We then switched to usual python dictionaries, of which we specially took advantage of Counter() and defaultdict(Counter)
# We have also used Pandas Dataframe, but it was just to display the confusion matrix in a presentable form.
# Everything else was implemented using python dictionaries and other usual data types.
#
# 2) In order to reduce the word count, we performed the following word sanitization:
# Word stripping, converted words to lower, filtered out words whose
# frequency was less than 20 and also removed the stop words

# Show accuracies as a function of fraction
# Cases when fraction is 1 (fully supervised), 0 (unsupervised), or low (e.g. 0.1).
# Pls PFA the Readme file for Accuracies over different fraction and their confusion Matrix


import copy
import sys
import os
import collections
from collections import defaultdict, Counter
import math as m
import time
import pickle
import re
import random
from Queue import PriorityQueue
import pandas as pd
import string

# Global Variables
all_words = defaultdict(int)
all_words_list = defaultdict(int)
all_topics_list = []
stop_words_dict = defaultdict(int)
topic_priors = Counter()
topic_word_likelihood = defaultdict(Counter)
labelled_topic_doc_posteriors = []
unlabelled_topic_doc_posteriors = []
unlabelled_documents_text = []
file_list = []
full_data = defaultdict(int)
topic_priors_backup = Counter()
topic_word_likelihood_backup = defaultdict(Counter)


# directory = "E:\\Rahul\\IUB Education\\Fall 2016\\AI (David Crandell)\\Assignments\\Assignment-4\\rrsampat-vsureshb-anveling-a4\\part2"

# Writes all the dicts to the text files
def write_all_dict_to_files(file_name):
    write_data_to_file(file_name)


# Writes a particular dict to the text files
def write_data_to_file(file_name):
    print("Writing files to pickle...")
    s_time = time.time()
    global file_list
    print "All topic priors", len(topic_priors)
    print "topic_word_likelihood", len(topic_word_likelihood)
    print "all_words_list", len(all_words_list)
    print "all_topics_list", len(all_topics_list)
    file_list.append(topic_priors)
    file_list.append(topic_word_likelihood)
    file_list.append(all_words_list)
    file_list.append(all_topics_list)
    #file_list.append(all_words)

    pickle.dump(file_list, open(file_name, "wb"))
    print "Time to write files: ", time.time()-s_time
    # pickle.dump( dictname, open( filename, "wb" ) )
    # with open(filename, 'w') as f:
    #     pickle.dump([nb_file_count_dict, nb_word_count_dict, nb_binary_prob_dict, nb_cont_prob_dict], f)


# Reads all the dicts to the text files
def reads_all_dict_to_files(file_name):
    read_data_from_file(file_name)


# Reads the dict from the text files and stores in the dict
def read_data_from_file(file_name):
    print("Reading files from pickle...")
    s_time = time.time()
    global file_list, all_words, topic_priors, topic_word_likelihood, all_words_list, all_topics_list
    file_list = pickle.load(open(file_name, "rb"))
    topic_priors = file_list[0]
    topic_word_likelihood = file_list[1]
    all_words_list = file_list[2]
    all_topics_list = file_list[3]
    print "All topic priors", len(topic_priors)
    print "topic_word_likelihood", len(topic_word_likelihood)
    print "all_words_list", len(all_words_list)
    print "all_topics_list", len(all_topics_list)
    print "Time to read files: ", time.time()-s_time
    # all_words = file_list[4]


# Creates a stop words dict
def get_stop_words_dict():
    global stop_words_dict
    stop_words_list = ['a', 'able', 'about', 'above', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available', 'away', 'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but', 'by', 'c', 'ca', 'came', 'can', 'cannot', "can't", 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come', 'comes', 'contain', 'containing', 'contains', 'could', 'couldnt', 'd', 'date', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', 'done', "don't", 'down', 'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'except', 'f', 'far', 'few', 'ff', 'fifth', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit', 'however', 'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inc', 'indeed', 'index', 'information', 'instead', 'into', 'invention', 'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself', "i've", 'j', 'just', 'k', 'keep\tkeeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 'liked', 'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'now', 'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven', 'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should', "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly', 'since', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure\tt', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 'therefore', 'therein', "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon', "there've", 'these', 'they', 'theyd', "they'll", 'theyre', "they've", 'think', 'this', 'those', 'thou', 'though', 'thoughh', 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', "'ve", 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', 'wasnt', 'way', 'we', 'wed', 'welcome', "we'll", 'went', 'were', 'werent', "we've", 'what', 'whatever', "what'll", 'whats', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who', 'whod', 'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'willing', 'wish', 'with', 'within', 'without', 'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y', 'yes', 'yet', 'you', 'youd', "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves', "you've", 'z', 'zero', 'amount', 'bill', 'bottom', 'call', 'cant', 'con', "couldn't", 'cry', 'de', 'describe', 'detail', 'eleven', 'empty', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'forty', 'front', 'full', 'interest', 'keep', 'mill', 'mine', 'move', 'no one', 'serious', 'side', 'sincere', 'sixty', 'system', 'ten', 'thick', 'thin', 'third', 'three', 'top', 'twelve', 'twenty', 'well', 'will', "it's"]
    for word in stop_words_list:
        stop_words_dict[word] = 1


# Sanitizes the word for NB
def nb_sanitize_word(word):
    word = word.strip()
    #replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    #word = word.translate(replace_punctuation)
    word = word.lower()
    # word = word.translate(None, string.punctuation)
    return word


# Finds and prints the top & least 10 spam associated words
def get_distinctive_words():
    pq_dict = defaultdict(int)
    total_docs = sum(topic_priors.values())
    # freq_threshold = 2.0/(50.0 * len(all_topics_list))
    freq_threshold = 0.001
    for topic in topic_priors:
        topic_words_probability = PriorityQueue()
        for word in topic_word_likelihood[topic]:
            if topic_word_likelihood[topic][word] > freq_threshold:
                # print "True", topic, word, topic_word_likelihood[topic][word]
                numerator = max(1E-6, topic_word_likelihood[topic][word])
                # numerator /= max(1E-6, float(sum(topic_word_likelihood[topic].values())))
                denominator = 0
                for other_topics in topic_priors:
                    denominator += max(1E-6, topic_word_likelihood[other_topics][word])
                    # denominator += max(1E-6, topic_word_likelihood[other_topics][word]) * max(1E-6, 1.0 * topic_priors[other_topics]/total_docs)
                    # /max(1E-6, float(sum(topic_word_likelihood[other_topics].values())))
                word_distinctive_prob = m.log(numerator) - m.log(denominator)
                topic_words_probability.put((-word_distinctive_prob, word))
        print "PQ", topic, topic_words_probability.qsize()
        pq_dict[topic] = topic_words_probability
        # print "PQ Dict",topic,  pq_dict[topic].queue()
    return pq_dict


# Writes the 20 words to the file:
def write_distinctive_words(pq_dict):
    print("Saving Distinctive Words...")
    print "pq_dict", pq_dict
    f = open("distinctive_words.txt", "w")
    f.write("20 distinctive words by topics:\n")
    topic_count = 0
    print "topic priors", topic_priors
    for topic in topic_priors.keys():
        # print "pq_dict", pq_dict[topic].qsize()
        # print "pq_dict", pq_dict[topic].queue()
        topic_count += 1
        f.write(str(topic_count) + ". " + topic + ":\n")
        for x in range(1, 11):
            f.write(str(x) + ". " + pq_dict[topic].get()[1] + "\n")
        f.write("\n\n")
    f.close()


# Returns a list of words in a given email file
def create_doc_string(file_path):
    try:
        with open(file_path) as f:
            # f = open(file_path)
            email_string = f.read().split()
            return email_string
            # email_string = f.read()
            # return re.findall('[A-Za-z0-9]+', email_string)
            # f.close()
    except IOError:
        print("Cannot read from file:", file_path)
        return 0


# Sets the posterior probability of a doc to its topic
def set_doc_topic_posterior(label_known, subdir, doc_string):
    global labelled_topic_doc_posteriors, unlabelled_topic_doc_posteriors
    l_posterior = []
    posteriors = Counter()
    for topic in topic_priors.keys():
        posteriors[topic] = 0
    if label_known:
        posteriors[subdir] = 1
        labelled_topic_doc_posteriors.append((posteriors, doc_string))


# Reads the training data from a given file
def read_training_data(directory_path, fraction):
    print("Reading the training data and creating a vocabulary...")
    global topic_priors, topic_word_likelihood, all_words, unlabelled_documents_text, topic_priors_backup, topic_word_likelihood_backup
    fraction = fraction if fraction != 0 else 0.001
    doc_count = 0
    for doc in full_data:
        doc_count += 1
        subdir = full_data[doc]['topic']
        doc_string = full_data[doc]['text']
        label_known = coin_flip(fraction)
        # if fraction < 1.0:
        # Sets the document to topic probability of 1 for known documents and 0 for unknown documents
        set_doc_topic_posterior(label_known, subdir, doc_string)
        if label_known:
            # Updating topic probabilities
            topic_priors[subdir] += 1
            for word in doc_string:
                word = nb_sanitize_word(word)
                if word in all_words_list:
                    topic_word_likelihood[subdir][word] += 1
        else:
            unlabelled_documents_text.append((subdir, doc_string))
    topic_priors_backup = copy.deepcopy(topic_priors)
    topic_word_likelihood_backup = copy.deepcopy(topic_word_likelihood)


# Gets the posterior probability for each doc belonging to each class
def get_class_posteriors(doc_string, t_priors, t_w_likelihood, max_iterations, current_iteration):
    # print "Getting class posteriors..."
    # E-Step: Returns the class that maximizes the posterior
    posteriors = Counter()
    for topic in t_priors:
        p_topic = m.log(max(1E-6, t_priors[topic]))
        # topic_total_words = float(sum(t_w_likelihood[topic].values()))
        for word in doc_string:
            word = nb_sanitize_word(word)
            if word in all_words_list:
                # p_topic += m.log(max(1E-6,t_w_likelihood[topic][word])) - m.log(topic_total_words)
                p_topic += m.log(max(1E-6, t_w_likelihood[topic][word]))
        posteriors[topic] = p_topic
    total = sum(posteriors.values())
    if total == 0:
        return posteriors
    for topic in posteriors.keys():
        posteriors[topic] /= total

    # Making the maximum probability to 1 after the last iteration
    # if current_iteration == max_iterations - 1:
    #     topic_max = max(posteriors, key = lambda x: posteriors[x])
    #     # for topic_priors in posteriors.keys():
    #     #     posteriors[topic_priors] = 1 if topic_priors == topic_max else 0
    #     #
    #     posteriors = Counter()
    #     posteriors[topic_max] = 1
    #
    return posteriors


# Predicts the labels of the unlabelled examples in the training set
def predict_unlabelled_train_topics(iterations):
    print "Predicting the labels for the unlabelled training data set..."
    global topic_priors, topic_word_likelihood, unlabelled_topic_doc_posteriors
    for i in range(iterations):
        s_time = time.time()
        print "Loop i :", i
        unlabelled_topic_doc_posteriors = []
        # num_correct = 0
        normalize_likelihood()
        # num_unlabelled_docs = len(unlabelled_documents_text)
        # num_classified = 0
        s_time2 = time.time()
        for doc in unlabelled_documents_text:
            # predicted_class_label = naive_bayes(doc[1], topic_priors, topic_word_likelihood)
            # num_classified += 1
            # if predicted_class_label == doc[0]:
            #     num_correct += 1
            unlabelled_topic_doc_posteriors.append \
                ((get_class_posteriors(doc[1], topic_priors, topic_word_likelihood, iterations, i), doc[1]))
        print("Time taken for NB + Get_Class_Posterior: ", time.time()-s_time2)
        s_time2 = time.time()
        # print "Training set Accuracy: ", float(100.00 * num_correct) / num_classified
        topic_priors, topic_word_likelihood = recalculate_prior_likelihood \
            (labelled_topic_doc_posteriors, unlabelled_topic_doc_posteriors)
        print("Time taken for Recalculating Prior Likelihood: ", time.time()-s_time2)
        print("Time taken for " + str(i) + " loop", time.time() - s_time)


# Recalculates the priors for the training dataset
def recalculate_prior_likelihood(labelled_posterior, unlabelled_posterior):
    print "Recalculating Prior Likelihoods..."
    # print "unlabelled_topic_doc_posteriors:", unlabelled_topic_doc_posteriors
    # M-Step: Use the E-Step's classification to get the priors, likelihood
    priors = copy.deepcopy(topic_priors_backup)
    likelihood = copy.deepcopy(topic_word_likelihood_backup)
    print "Unlabelled Size: ", len(unlabelled_posterior)

    for (posterior, doc_string) in unlabelled_posterior:
        for topic in posterior.keys():
            priors[topic] += posterior[topic]
            for word in doc_string:
                word = nb_sanitize_word(word)
                likelihood[topic][word] += posterior[topic]
                # Trying to add a new word to which might come in the unlabelled training document
                all_words_list[word] += 1
    return priors, likelihood


# Updates the all topics dict
def update_all_topics(directory_path):
    print "Updating All Topics List..."
    global all_topics_list, topic_priors, topic_word_likelihood
    for dir in os.walk(directory_path):
        for subdir in dir[1]:
            all_topics_list.append(subdir)
            topic_priors[subdir] = 0
            topic_word_likelihood[subdir] = Counter()


# Updates all words initially by iterating through all the training docs
# dataset = "train"
def update_all_words(directory_path):
    print "Updating All Words Dict..."
    s_time = time.time()
    global all_words_list, all_words, full_data
    # for dir in os.walk(directory_path):
    #     for subdir in dir[1]:
    global_doc_counter = 0
    for subdir in os.listdir(directory_path):
        if subdir in all_topics_list:
            subdir_path = directory_path + os.sep + subdir + os.sep
            for doc in os.listdir(subdir_path):
                global_doc_counter += 1
                doc_dict = defaultdict(int)
                doc_dict['topic'] = subdir
                doc_dict['fname'] = doc
                file_path = subdir_path + doc
                doc_string = create_doc_string(file_path)
                doc_dict['text'] = doc_string
                doc_dict['label'] = subdir
                # doc_dict['word_count'] = 0
                for word in doc_string:
                    word = nb_sanitize_word(word)
                    if word not in stop_words_dict:
                        all_words[word] += 1
                full_data[global_doc_counter] = doc_dict
    print("Time to make all words Dict : ", time.time()-s_time)


# Updates all words global list, but skips the words whose frequency is below word frequency parameter
def update_all_words_list(frequency_threshold):
    print "Updating All Words List..."
    s_time = time.time()
    global all_words_list
    for word, frequency in all_words.items():
        if frequency > frequency_threshold:
            all_words_list[word] = frequency
    print("Time to filter the words : ", time.time()-s_time)

# Train the data
def nb_train_data(directory_path, fraction):
    print("Training Naive Bayes Classifier...")
    print("Thank you for waiting...\nWe appreciate your patience...")
    get_stop_words_dict()
    directory_path = directory_path + os.sep  # + dataset
    update_all_topics(directory_path)
    update_all_words(directory_path)
    update_all_words_list(0)
    read_training_data(directory_path, fraction)
    if fraction < 1:
        predict_unlabelled_train_topics(5)
    normalize_likelihood()
    pq_dict = get_distinctive_words()
    write_distinctive_words(pq_dict)


# Normalizing the likelihood
def normalize_likelihood():
    global topic_priors, topic_word_likelihood
    for topic in topic_priors.keys():
        total_words = float(sum(topic_word_likelihood[topic].values()))
        for word in topic_word_likelihood[topic].keys():
            topic_word_likelihood[topic][word] /= total_words


# Reads the test data from the given directory
# dataset = "test"
# subdir = "spam"/"notspam"
def nb_test_data(directory_path):  # , prob_dict, nb_type, file_name):
    print "Testing the test data set..."
    get_stop_words_dict()
    test_file_count = 0
    test_correct_count = 0
    nb_confusion_matrix, all_top_list = create_confusion_matrix()

    directory_path = directory_path + os.sep  # + dataset
    # for subdir in os.listdir(directory_path):
    #     if subdir in subdir_list:
    for director in os.walk(directory_path):
        for subdir in director[1]:
            subdir_path = directory_path + os.sep + subdir + os.sep
            for doc in os.listdir(subdir_path):
                file_path = subdir_path + doc
                doc_string = create_doc_string(file_path)
                test_file_count += 1
                # nb_confusion_matrix = naive_bayes(subdir_path + email, train_prob_spam, train_prob_not_spam, subdir, prob_dict, nb_confusion_matrix)
                # test_correct_count += naive_bayes(subdir_path + email, train_prob_spam, train_prob_not_spam, subdir, prob_dict, nb_confusion_matrix)
                doc_topic = naive_bayes(doc_string, topic_priors, topic_word_likelihood)  # , nb_confusion_matrix)
                if doc_topic == subdir:
                    test_correct_count += 1
                nb_confusion_matrix[subdir][doc_topic] += 1
                nb_confusion_matrix[subdir]["Actual Count"] += 1

    print "Test Correct Count:", test_correct_count
    print "Testing Doc Count:", test_file_count
    print_confusion_matrix(nb_confusion_matrix, all_top_list)
    return calc_accuracy("nb_confusion_matrix", test_correct_count, test_file_count)
    # return (100.0 * test_correct_count)/test_file_count


def naive_bayes(doc_string, t_priors, t_w_likelihood):  # , nb_confusion_matrix):
    max_class = (-1E6, '')
    for topic in t_priors:
        p_topic = m.log(t_priors[topic])
        # topic_total_words = float(sum(t_w_likelihood[topic].values()))
        for word in doc_string:
            word = nb_sanitize_word(word)
            if word in all_words_list:
                p_topic += m.log(max(1E-6, t_w_likelihood[topic][word]))  # - m.log(topic_total_words)
        if p_topic > max_class[0]:
            max_class = (p_topic, topic)

    return max_class[1]
    # return nb_confusion_matrix


# Calculates accuracy based on the given confusion matrix
def calc_accuracy(confusion_matrix, test_correct_count, test_file_count):
    return (100.0 * test_correct_count) / test_file_count


# Creates confusion matrix dictionary
def create_confusion_matrix():
    confusion_matrix = collections.OrderedDict()
    print "All topic prior", len(topic_priors)
    all_top_list = topic_priors.keys()
    for topic in all_top_list:
        confusion_matrix[topic] = collections.OrderedDict()
        for all_topics in all_top_list:
            confusion_matrix[topic][all_topics] = 0
        confusion_matrix[topic]["Actual Count"] = 0
    print "CM", confusion_matrix
    return (confusion_matrix, all_top_list)


# Print confusion matrix
def print_confusion_matrix(confusion_matrix, all_top_list):
    print "Confusion Matrix"
    print "\t Model Results"
    all_top_list.append("Actual Count")
    confusion_list = [[all_top_list[y] for x in range(len(all_top_list) + 1)] for y in
                      range(len(all_top_list) - 1)]
    confusion_header = ["Actual\Model"] + all_top_list
    for i in range(len(confusion_header) - 2):
        for j in range(1, len(confusion_header)):
            confusion_list[i][j] = confusion_matrix[all_top_list[i]][confusion_header[j]]
    cm = pd.DataFrame(confusion_list, columns=confusion_header)
    print cm.to_string(index=False)
    print "\n"


# Returns a coin flip probability based on the given fraction
def coin_flip(fraction):
    return random.random() >= 1 - fraction


# Called to initialize the program
def main(mode, directory, file_name, fraction):
    # technique = 'bayes'
    if mode == "train":
        nb_train_data(directory, fraction)
        write_all_dict_to_files(file_name)
    elif mode == "test":
        reads_all_dict_to_files(file_name)
        accuracy = nb_test_data(directory)
        print ("Naive Bayes Average Accuracy", float(accuracy))


if __name__ == '__main__':
    start_time = time.time()
    mode = str(sys.argv[1])
    directory = str(sys.argv[2])
    file_name = str(sys.argv[3])
    fraction = float(sys.argv[4])
    file_name += ".p"
    main(mode, directory, file_name, fraction)

    end_time = time.time()
    print ("Total Time", end_time - start_time)
