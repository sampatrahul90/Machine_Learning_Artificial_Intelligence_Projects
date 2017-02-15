NOTE: THIS PROGRAM IS COMPATIBLE WITH JUST 3.4 OR HIGHER VERSION!!!
KINDLY RUN IT ON A COMPATIBLE PYTHON VERSION
MAY GIVE ERRORS/INCORRECT RESULTS ON PYTHON 2.7

This is the main file to run the program.
Also there are 3 other files (used for decision tree).
Please keep them in the same folder as well.
The other 3 files are as follows:
1) functions_dt.py 
2) main_dt.py
3) Node.py

#Part 1a - Naive Bayes Classifier

Formulation of the problem:
Part 1a of the problem was done by implementing a Naive Bayes Classifier
Data Pre-processing - Word Sanitization - I tried sanitizing the text in various ways like stripping,
converting to lower, removing punctuations and stops words,only to later on realize that the best accuracy
for Naive Bayes was achieved by not doing anything at all.

I am using pickle package of python to store files in filename.p format.
The object/variable can directly be retrieved from this file and can be stored in a variable directly.

Note: Please run the code to get the Accuracy Results, Confusion Matrix and the Top & Least Spam Words

Working:
Does the below procedure for both Binary and Continuous Classifier

1) Training the classifiers:
a) Creates Bag of words for both spam and non-spam documents and stores in a dict.
b) Calculates prior probabilities & likelihood for them
c) Stores the data in the pickle file
d) Print the Top & Least 10 words for Binary and Continuous

I am calculating the Top & Least Spam words with the below formula:
P(S=1|w_i) = P(w_i|S=1)/P(w_i|S=1) + P(w_i/S=0)
I am ignoring P(S=1), since its a constant for each topic and we have to find the top words per topic and
not across topics.

2) Testing the classifier:
a) Reads the trained data from the pickle file and stores in the respective variables.
b) Runs the learned Decision Tree on the test data
c) Creates and prints the confusion matrix
d) Creates and prints the accuracy

Note: Please run the code to get the Accuracy Results, Confusion Matrix and the Top & Least Spam Words

Naive Bayes Binary Accuracy 98.39467501957714
Naive Bayes Continuous Accuracy 97.41581832419733
Naive Bayes Average Accuracy 97.90524667188723

As we can see, the Binary Features for Naive Bayes works slightly better than with Continuous Features
Also, word sanitization hampers the performance/accuracy of the classifier


#Part1b - Decision Tree

Formulation of the problem:
Part 1a of the problem was done by implementing a Naive Bayes Classifier
Data Pre-processing - Word Sanitization - For DT, I did sanitize the words by stripping
and converting them to lower case.
It actually improved the accuracy

Working:
Does the below procedure for both Binary and Continuous Classifier
1) Training the classifiers:
a) Creates Bag of words (and simultaneously sanitizes them as well )for both spam and non-spam documents
and stores in a dict
b) Filters the top words (whose frequency is more than a certain threshold-passed as a
parameter to update_all_words_list() function). For now, I am considering the words whose frequency is more than 300.
c) Stores the class labels and above selected most frequent words as their features as a training data set
into a csv file (for for Binary and Continuous features separately),which are later used to build the tree.
The csv format - Saves Emails as rows and Words as feature columns. The first column is the class label.
d) Writes all the variables in a pickle file
e) Builds and Prints the Decision Tree for both Binary and Continuous Features

2) Testing the classifier:
a) Reads all the variables from a pickle file
b) Reads the test data and creates a similar csv file as created for training data above
c) Runs the Decision Tree Classifier to predict the label of the test
d) Calculates and prints the accuracy and prints the confusion matrix for both type of features

#Design Decisions:
I am converting the data to a csv and then using it to build the tree

I have used Pandas for storing the data in the 'Dataframe' data structure and some of its basic
functions of it to find row/column count and filtering rows/column.
We haven't used any advanced functionality of it.

We are calculating Information Gain = (Entropy before split-Entropy after split) and correspondingly
choosing the best attribute and its respective value

Also for continuous values, if the distinct values is more than 4, then we divide the whole list into 2 lists
and take the medium
of the respective sub lists for better splitting and efficiency


As we can see, the Continuous Features for Decision Tree works slightly better than with Binary Features
Also word sanitization - stripping, converting to lower, removing punctuations and stops words. works in favor of
Decision Tree Classifier and improves its accuracy, which is not the case with Naive Bayes Classifier.


#Naive Bayes VS Decision Tree for Spam Classification:
I would say that Naive Bayes works better for the Spam Classification Task over Decision Tree algorithm
for the following reasons:
1) Greater Accuracy (There is a slight difference, but it may depend on the data set as well)

2) Naive Bayes performs much faster as compared to Decision Tree:
(NB- ~20-30 secs) vs (DT-300-400 secs)
which makes Decision Tree almost 10 times slower, and the difference would exponentially increase with the size of
the data set

3) If all the words/features from the training data set are taken into consideration for the Decision Tree, then it
will become even more slower and might also over fit the data. We do not face such issues with Naive Bayes



The program should accept command line arguments like this:

    ./spam mode technique dataset-directory model-file

where 

mode is either test or train, 

technique is either bayes or dt, dataset-directory is a directory containing two subdirectories named spam and notspam, 
each filled with documents of the corresponding type, and 

model-file is the filename of your trained model. 

In training mode, dataset-directory will be the training dataset and the program should write the trained model to model-file, 

and in testing mode dataset-directory will be the testing dataset and your program should read its trained model from model-file.
