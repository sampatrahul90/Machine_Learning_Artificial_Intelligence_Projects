How does it work ?

The computer player tries to generate all possible board states by placing the current piece in all possible columns
and rotations, and then placing the next piece in all possible columns/rotations. Then it evaluates which
is the best possible move based on certain features and its weight(Explained below).
It picks the one with the highest evaluation value. The computer
player then plays this move on the actual board with the chosen column and rotation of the current piece.
This process is repeated for each piece.


Explanation for Features and Weights:
 The Features is a metric which actually decides which is an ideal board that we want to see. We referred to certain
 articles and papers who have done analysis on the different features and we decided to pick 6 features \
 out of them for our implementation here. We have also added weights to each of these features based on the explanation
  in the paper as well as several tests that we ran.

  Feature 1 - Aggregate Height: Sum of heights of each column is a negative feature.
  
  Feature 2 - Complete Lines: Number of lines complete lines is a positive feature.
  
  Feature 3 - Bumpiness: sum of difference in height between each columns is a negative feature.
  
  Feature 4 - Holes count: Number of spaces with atleast a filled spot above it in the same column is a negative feature.
  
  Feature 5 - Altitude Delta: Different is height between the tallest and shortest column is a negative feature.
  
  Feature 6 - Weighted Holes: Sum of the holes weighed by the row they are located in is a negative feature.

  General Formula for Evalution:
  Feature1 * Weight1 + Feature2 * Weight2 +.......+ FeatureN * WeightN

    Here, the Weight(N) can be negative or positive based on whether it is a negative or positive feature.

How to Run ?

    python ./tetris.py computer animated

References:
    1. http://www.cs.uml.edu/ecg/uploads/AIfall10/eshahar_rwest_GATetris.pdf
    2. https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
