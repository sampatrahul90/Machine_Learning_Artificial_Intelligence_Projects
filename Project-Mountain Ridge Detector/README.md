
Ridge Detection using Markov Chain Monte Carlo


Abstractions:

I have formulated the problem as a hidden markov model, where the parameters are as follows:
1.) - State - Each state can be a representation of the column, where gradient is the observed variable and we are trying to determine the probability of row being selected

2.) - Observed Variables - The gradient data of the column
2.) - Hidden Variables - Here in we are considering the hidden variables to be the row numbers.

3.) - Transition Probabilities - This is the probabilities which help us determine a given state, based on its neighbours. For us it depends upon the previous and the next state. - We look to maximise this factor, to ensure smoothness i.e. keep in mind that the next state is not abrubptly away from the current and previous. Hence we focus our search to nearest possible neighbours 

4.) - Emission Probabilites - This in our problem, is the variable that connects hidden and observed states, this is clearly visible to us. 
- We normalise this using the sum of inspected gradients by, dividing the given row gradient by the sum.



1.) Part I - This is simple and is implemented in best_edge_list(edge_length) - The idea was to simply pick the best gradient there is and use it for the corresponding column. - Keep doing this for each row and you shall have an output. - Well, it is not completely efficient, but it is a start.


2.) Part II -
a) In this problem, we are applying MCMC-Gibbs sampling on the HMM and then taking the mode of the row points in each column

b) I am iterating for as many times as the number of columns in the image

c) I am also restricting the row number to be explored in a given column based on the previous and next row number

d) As a part of probability calculation, I have considered the (prev_row point distance , next_row_point distance the grad_value)

e) Below is the formula I have used
		temp_current_pos_prob = (float(1) / (prev_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
				   float(2) / (next_relative_dist + 1)) * (2.0 * (total_rows - row) / (total_rows))
		where maxc is the max gradient in the entire image


3.) Part III -
a) In this problem, we are applying MCMC-Gibbs sampling on the HMM and then taking the mode of the row points in each column

b) For this part, I am just doing a few iterations starting from the given user point

c) I am also restricting the row number to be explored in a given column based on the previous and next row number

d) Here I am increasing the grad value of 20 rows above and beow the given gt_row (in each columns), so as to increase their probability of getting chosen


-Problems faced:-
Since there was no training data provided, the formulation of the general probability distribution (which would work over all the images)
required a lot of trial and error



    To run: 
      python mountain.py [input image path/name] [output image name] X Y

      Eg:
      python mountain.py test_images/mountain.jpg output.jpg 152 171

    mountain.py - Script
    test_images/mountain.py - Test Image

    output.jpg - Output Image name

    X coordinate on the ridge in the image

    Y Coordinate
