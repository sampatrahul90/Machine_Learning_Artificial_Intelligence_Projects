#Q2 - Topic Classification

In this problem, we had to classify the documents among the given 20 topics, the catch being we can observe
the labels of only a fraction of the training set.

NOTE: THIS CODE IS PYTHON 2.7 COMPATIBLE. PLEASE DO NOT RUN IT ON PYTHON 3 VERSION!!


Due to a huge data set, the code takes 5+ mins to execute completely.

We have used the Naive Bayes algorithm along with Expectation Maximization to first predict the topics/labels
of the unknown documents from the training data set itself.
We re-iterate the Expectation Maximization to a point where the probabilities converge & label the unlabelled data set
Once the training data set is completed labelled, we calculate the priors to label the test data set.

We are doing that as follows:

1) We have first divided the data set into 2 parts-Labelled and Unlabelled

2) In Expectation Maximization, we are then calculating the priors and the likelihood using just the labelled
trained data set

3) Now using these priors, we are calculating the probability of the labels of the documents in unlabelled
training data set

4) Now for a document in an unlabelled data set, we get some probability of it being in each of the topics.

5) Now taking these probabilities of the unlabelled data set, along with the previous probabilities of the
labelled dataset,
we are recalculating the priors.

6) We are doing this for 5 iterations, and expecting the probabilities of the labels to converge by then.

7) After 5 iterations, we are taking the maximum probability of each document being in a class/topic and
assigning the corresponding topic to that document.

8) Once the training data set is completed labelled (each document belonging to just one topic
with a probability of 1), we re-calculate the priors one last time.

9) We are then using these priors and likelihood to predict the labels of the test data set.

#Design Decisions & Challenges:

1) It was really difficult to come with a program that would handle such a large data set.
I tried creating a NxN numpy array for storing the table with the information of word vs document vs topic presence,but the matrix
was too large and hence was really slow at performing the task.
We then switched to usual python dictionaries, of which we specially took advantage of Counter() and defaultdict(Counter)
We have also used Pandas Dataframe, but it was just to display the confusion matrix in a presentable form.
Everything else was implemented using python dictionaries and other usual data types.


2) In order to reduce the word count, we performed the following word sanitization:
Word stripping, converted words to lower, filtered out words whose
frequency was less than 20 and also removed the stop words


Show accuracies as a function of fraction
Cases when fraction is 1 (fully supervised), 0 (unsupervised), or low (e.g. 0.1).
Pls PFA the Readme file for Accuracies over different fraction and their confusion Matrix

# How to run:

The program should accept command line arguments like this:

    ./topics mode dataset-directory model-file [fraction]

where 

mode is either test or train, 

dataset-directory is a directory containing directories for each of the
topics (you can assume there are exactly 20)

model-file is the filename of your trained model. 

In training mode, an additional parameter fraction should be a number between 0.0 and 1.0 indicating the fraction of
labeled training examples that your training algorithm is allowed to see.
