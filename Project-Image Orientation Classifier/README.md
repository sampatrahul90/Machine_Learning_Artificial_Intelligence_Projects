We tried 3 different machine learning algorithms to identify the orientation of an image.

#Part 1: k-nearest neighbours

kNN - Takes ~ 8-9 mins to run

kNN being a lazy algorithm does not learn anything during training, but instead just stores the entire train data
set. It then finds the distance b/w each test instance and each train distance and assigns the majority of the train
class label of its "nearest" k neighbors.
An optimal value of k varies from problem to problem and we need to find that by experimenting.

Part 1:

We have done the following for k-means:

1) As a part of training, we converted the input image data from text files and stored them as numpy array

2) Then on the test data, we simply calculated the euclidean distance between each test instance and train instance
and then return the majority of the train class label of its "nearest" k neighbors.

3) After experimentation we found k=50 to work the best with the provided data set.

Note: Kindly see the report for accuracies over different values of k. And the confusion matrix when k = 50

Design Decisions:

Initially we used normal python list to store the train and test data and calculate the distance b/w them.
But since the data was huge, it took almost ~1.5 hrs for knn to run on the full data set
We then used np.array to store the train and test data and calculate the distance b/w them.
This gave us a huge performance improvement, with the test time of ~8 mins on the full data set.

Note: kNN takes ~8 mins to run on the provided data set. Time depends on the train and test data set size. Hence
it might differ on other data sets.


#Part 2: AdaBoost

AdaBoost - Takes ~ 15 seconds to run (Best Stump Count = 100)

The basic idea of AdaBoost algorithm is to create a bunch of weak classifiers which in combination can make a strong
decision.
We have done the following for adaBoost:

1) We are taking 2 random points to decide whether a file has been classified correctly or incorrectly.

2) While training, for each orientation we are creating n stumps (n = stump_count) and storing their classifier weight
and their 2 random points

3) While testing, we run these classifiers for each/per orientation. If a file a classified correctly, we add its
alpha and subtract otherwise. Since this is being done for each orientation, we get 4 different alphas values, one
for each orientation.

4) Now we compare for these alphas each orientation and consider the corresponding orientation of the highest alpha

5) After experimentation we found stump_count = 100 to work the best with the provided data set. We need to use a high
stump value since we taking completely random pairs of points and not finding the best ones.
The accuracy rises quickly at first, but then slows down once the stump size becomes greater than 25.
It gets consistent and stops improving much after stump = 100.
Also, since we are taking random points, the accuracies may vary (by ~2-3%) even for the same stump value.


Note: Kindly see the report for accuracies over different values of stump_count. And the confusion matrix when
stump_count = 100

Note: We are using pandas.Dateframe just to display the confusion matrix, since it provides good alignment like that
of a table. Pandas hasn't been used for any other purpose.


#Part 3: Neural Network

Neural Network: Takes ~ 5-6 mins to run (Best Hidden Count = 25)
You may try with smaller hidden count for better performance (Hidden Count = 10 ; T = ~200 seconds)

We have done the following for Neural Network:

1) We have implemented a single layer neural network with 100 iterations

2) We are initializing the value of the learning rate (alpha) to be 0.9.
Learning Rate (alpha) = 0.9 (I am increasing its power by 1 after every 10 iterations
eg. After 20 iteration - alpha = (0.9)^2 ; after 30 iterations - alpha = (0.9)^3 and so on...)

3) We are using stochastic  gradient descent as our activation function

4) After experimentation we found hidden_count = 50 to work the best with the provided data set. We need to use a high
stump value since we taking completely random pairs of points and not finding the best ones.

Note: Kindly see the report for accuracies over different values of hidden_count. And the confusion matrix when
hidden_count = 25


#Design Decisions

1) Firstly, to improve performance, we tried not using all the 192 features for Neural Network. Instead, we were
taking an average of 2 consecutive neighbors and storing as one value. This made our input feature vector for each
train/test file to be of 96 instead of 192.
This was done with an intuition that not much changes are captured in consecutive pixels and hence the information
loss is minimal.

This trick gives us a little performance improvement, but sacrificed ~3-5% of the accuracy.
Hence, reverted the change back and used full set of features.

2) Also to avoid huge calculations, we are normalizing both the train and test input vectors.

3) Learning Rate (alpha) = 0.9 (Initially) (I am increasing its power by 1 after every 10 iterations
eg. After 20 iteration - alpha = (0.9)^2 ; after 30 iterations - alpha = (0.9)^3 and so on...)


Note: We are using pandas.Dataframe just to display the confusion matrix, since it provides good alignment like that
of a table. Pandas hasn't been used for any other purpose.


#How to run:

1) knn:
	
	python orient.py train_file.txt test_file.txt nearest
		
2) AdaBoost:

	python orient.py train_file.txt test_file.txt adaboost stump_count
	
3) Neural Network:
	
	python orient.py train_file.txt test_file.txt nnet hidden_count
	
4) Best Algorithm:

	python orient.py train_file.txt test_file.txt best model_file

It uses whichever algorithm and parameter settings we would recommend to give the best accuracy.

Final Report – Trend Analysis of these algorithms for Image Orientation Detection
