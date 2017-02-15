#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

# Abstractions
# I have formulated the problem as a hidden markov model, where the parameters are as follows:
# Observed Variables: The gradient data of the column
# Hidden Variables: The actual row point on the ridge
# Given assumption: The stronger the image gradient, the higher the chance that the pixel lies along the ridgeline.

# Part 1
# We have just taken the highest gradient value for each row and assumed no dependency as shown in the Bayes Net 1(b)
#
# Part 2
# In this problem, we are applying MCMC-Gibbs sampling on the HMM and then taking the mode of the row points in each column
# I am iterating for as many times as the number of columns in the image
# I am also restricting the row number to be explored in a given column based on the previous and next row number
# As a part of probability calculation, I have considered the (prev_row point distance , next_row_point distance the grad_value)
# Below is the formula I have used
# temp_current_pos_prob = (float(1) / (prev_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
#            float(2) / (next_relative_dist + 1)) * (2.0 * (total_rows - row) / (total_rows))
# where maxc is the max gradient in the entire image

#
# Part 3
# In this problem, we are applying MCMC-Gibbs sampling on the HMM and then taking the mode of the row points in each column
# For this part, I am just doing a few iterations starting from the given user point
# I am also restricting the row number to be explored in a given column based on the previous and next row number
# Here I am increasing the grad value of 20 rows above and beow the given gt_row (in each columns), so as to increase their probability of getting chosen
#

# Problems faced:
# Since there was no training data provided, the formulation of the general probability distribution (which would work over all the images)
# required a lot of trial and error


from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
from collections import defaultdict
import time
import random as r
from collections import Counter

initial_pos_dict = defaultdict(int)


# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    print (type(filtered_y))
    print (filtered_y ** 2)
    return filtered_y ** 2


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(max(y - thickness / 2, 0), min(y + thickness / 2, image.size[1] - 1)):
            image.putpixel((x, t), color)
    return image


# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]
gt_row = int(gt_row)
gt_col = int(gt_col)

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)
total_rows = edge_strength.shape[0]
total_columns = edge_strength.shape[1]


# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.

def maxContrast(edge_strength):
    return float(max([max(item) for item in edge_strength]))

maxc = maxContrast(edge_strength)

# Finds a new ridgeline starting from a given column for part 2
def find_ridge(new_col2):
    global initial_pos_dict
    ridge = []
    row_offset = int(total_rows / 8)
    for col in range(new_col2, -1, -1):
        current_pos_prob = 0
        prev_grad_row_pos = initial_pos_dict[col - 1] if col != 0 else -1
        next_grad_row_pos = initial_pos_dict[col + 1] if col != total_columns - 1 else -1
        row_range_begin = prev_grad_row_pos - row_offset if prev_grad_row_pos <= next_grad_row_pos else next_grad_row_pos - row_offset
        row_range_end = prev_grad_row_pos + row_offset if prev_grad_row_pos >= next_grad_row_pos else next_grad_row_pos + row_offset

        row_range_begin = 0 if row_range_begin < 0 else row_range_begin
        row_range_end = total_rows if (row_range_end > total_rows or prev_grad_row_pos == -1) else row_range_end
        for row in range(row_range_begin, row_range_end):
            current_grad_value = edge_strength[row][col]
            prev_relative_dist = abs(prev_grad_row_pos - row) if prev_grad_row_pos != -1 else 0
            next_relative_dist = abs(next_grad_row_pos - row) if next_grad_row_pos != -1 else 0

            temp_current_pos_prob = (float(1) / (prev_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
            float(2) / (next_relative_dist + 1)) * (2.0 * (total_rows - row) / (total_rows))

            if temp_current_pos_prob > current_pos_prob:
                initial_pos_dict[col] = row
                current_pos_prob = temp_current_pos_prob

    for col in range(new_col2 + 1, total_columns):
        current_pos_prob = 0
        prev_grad_row_pos = initial_pos_dict[col - 1] if col != 0 else -1
        next_grad_row_pos = initial_pos_dict[col + 1] if col != total_columns - 1 else -1
        row_range_begin = prev_grad_row_pos - row_offset if prev_grad_row_pos <= next_grad_row_pos else next_grad_row_pos - row_offset
        row_range_end = prev_grad_row_pos + row_offset if prev_grad_row_pos >= next_grad_row_pos else next_grad_row_pos + row_offset
        row_range_begin = 0 if row_range_begin < 0 else row_range_begin
        row_range_end = total_rows if (row_range_end > total_rows or prev_grad_row_pos == -1) else row_range_end
        for row in range(row_range_begin, row_range_end):
            current_grad_value = edge_strength[row][col]
            prev_relative_dist = abs(prev_grad_row_pos - row) if prev_grad_row_pos != -1 else 0
            next_relative_dist = abs(next_grad_row_pos - row) if next_grad_row_pos != -1 else 0

            ###--------------------New---------------------
            temp_current_pos_prob = (float(1) / (prev_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
            2.0 * (total_rows - row) / (total_rows))  # * (float(2)/(abs(next_relative_dist)+1))
            # temp_current_pos_prob = (float(1)/(prev_relative_dist+1)) * (float(current_grad_value) / maxc) * (2.0*(total_rows-row)/(total_rows)) * (float(2)/(abs(next_relative_dist)+1))

            if temp_current_pos_prob > current_pos_prob:
                initial_pos_dict[col] = row
                current_pos_prob = temp_current_pos_prob

    for key in range(len(initial_pos_dict)):
        ridge.append(initial_pos_dict[key])
    return ridge

# Finds a new ridgeline starting from a given column for part 3 (Given an input point)
def find_ridge_user(new_col2):
    global initial_pos_dict
    ridge = []
    row_offset = 10
    for col in range(new_col2, -1, -1):
        # Skipping the repositioning of the given column
        if col == gt_col:
            continue
        else:
            current_pos_prob = 0
            prev_grad_row_pos = initial_pos_dict[col - 1] if col != 0 else -1
            next_grad_row_pos = initial_pos_dict[col + 1] if col != total_columns - 1 else -1
            # Limiting the search to a limited range of rows
            row_range_begin = 0 if next_grad_row_pos - 0 < 0 else next_grad_row_pos - row_offset
            row_range_end = total_rows if (
            next_grad_row_pos + row_offset + 1 > total_rows or prev_grad_row_pos == -1) else next_grad_row_pos + row_offset + 1
            for row in range(row_range_begin, row_range_end):
                current_grad_value = edge_strength[row][col]
                prev_relative_dist = abs(prev_grad_row_pos - row) if prev_grad_row_pos != -1 else 0
                next_relative_dist = abs(next_grad_row_pos - row) if next_grad_row_pos != -1 else 0

                temp_current_pos_prob = (float(1) / (next_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
                10.0 * (total_rows - row) / (total_rows))
                # temp_current_pos_prob = (float(1)/(next_relative_dist+1)) * (float(current_grad_value) / maxc) * (10.0*(total_rows-row)/(total_rows)) * (float(2)/(prev_relative_dist+1))

                if temp_current_pos_prob > current_pos_prob:
                    initial_pos_dict[col] = row
                    current_pos_prob = temp_current_pos_prob

    for col in range(new_col2 + 1, total_columns):
        if col == gt_col:
            continue
        else:
            current_pos_prob = 0
            prev_grad_row_pos = initial_pos_dict[col - 1] if col != 0 else -1
            next_grad_row_pos = initial_pos_dict[col + 1] if col != total_columns - 1 else -1
            # Limiting the search to a limited range of rows
            row_range_begin = 0 if prev_grad_row_pos - row_offset < 0 else prev_grad_row_pos - row_offset
            row_range_end = total_rows if (
            prev_grad_row_pos + row_offset + 1 > total_rows or prev_grad_row_pos == -1) else prev_grad_row_pos + row_offset + 1
            for row in range(row_range_begin, row_range_end):
                current_grad_value = edge_strength[row][col]
                prev_relative_dist = abs(prev_grad_row_pos - row) if prev_grad_row_pos != -1 else 0
                next_relative_dist = abs(next_grad_row_pos - row) if next_grad_row_pos != -1 else 0

                temp_current_pos_prob = (float(1) / (prev_relative_dist + 1)) * (float(current_grad_value) / maxc) * (
                10.0 * (total_rows - row) / (total_rows))  # * (float(2)/(abs(next_relative_dist)+1))
                # temp_current_pos_prob = (float(1)/(prev_relative_dist+1)) * (float(current_grad_value) / maxc) * (10.0*(total_rows-row)/(total_rows)) * (float(2)/(abs(next_relative_dist)+1))

                if temp_current_pos_prob > current_pos_prob:
                    initial_pos_dict[col] = row
                    current_pos_prob = temp_current_pos_prob

    for key in range(len(initial_pos_dict)):
        ridge.append(initial_pos_dict[key])
    return ridge


# Finds the mode for each column from all the generated ridges
def convergeRidges(ridges):
    optimalRidge = []
    for col in range(total_columns):
        temp = []
        for row in range(len(ridges)):
            temp.append(ridges[row][col])
        count = Counter(temp)
        res = count.most_common(1)
        print "col", col, "res", res
        optimalRidge.append(res[0][0])
    print "*****************************************************************************************************************"
    return optimalRidge


# Creates initial dictionary with rows numbers of highest gradients
def create_init_pos_dict(edge_strength):
    global initial_pos_dict
    total_col_grad = 0
    # result = []
    for col in range(edge_strength.shape[1]):
        max_grad_value = 0
        row_pos = 0
        for row in range(edge_strength.shape[0]):
            total_col_grad += int(edge_strength[row][col])
            if int(edge_strength[row][col]) > max_grad_value:
                max_grad_value = int(edge_strength[row][col])
                row_pos = row
        # initial_pos_dict[col] = (row_pos, max_grad_value, total_col_grad, max_grad_value, float(1.0*max_grad_value/total_col_grad))
        initial_pos_dict[col] = row_pos
        total_col_grad = 0
    print "Total rows:", edge_strength.shape[0]
    print "Total columns:", edge_strength.shape[1]
    print initial_pos_dict
    # print initial_pos_dict[500][2]
    return initial_pos_dict


# Simple Bayes Net for Part 1
def Bayes_Net(edge_strength):
    create_init_pos_dict(edge_strength)
    result = []
    for x in range(edge_strength.shape[1]):
        maxi = 0
        res = 0
        for y in range(edge_strength.shape[0]):
            if int(edge_strength[y][x]) > maxi:
                maxi = int(edge_strength[y][x])
                res = y
        # print ("---------------")
        # print (maxi)
        result.append(res)
    print "Bayes Net Result: ", len(result), result
    return result


# HMM with MCMC-Gibbs Sampling for Part 2
def HMM_MCMC(edge_strength):
    global initial_pos_dict
    create_init_pos_dict(edge_strength)
    ridges = []
    # Generating random column samples for Gibbs sampling
    col_list = r.sample(xrange(0, total_columns), 200)
    n = 200 - len(col_list)
    if n > 0:
        col_list2 = r.sample(xrange(0, total_columns), n)
        col_list += col_list2
    #print "Before iteration Initial pos dict: ", initial_pos_dict
    iteration = 0
    for new_col2 in col_list:
        # iteration += 1
        # print "Iteration", iteration
        # iter_time = time.time()
        # print "new_col", new_col2
        # ridges = [find_ridge(col) for new_col2 in col_list]
        new_ridge = find_ridge(new_col2)
        ridges.append(new_ridge)
        #print "Time", time.time() - iter_time

    optimal_ridge = convergeRidges(ridges)
    return optimal_ridge


# HMM with MCMC-Gibbs Sampling given a User point for Part 3
def HMM_MCMC_user(edge_strength, gt_row, gt_col):
    global initial_pos_dict
    ridges = []
    # Updating the updating the edge strengths of certain rows above and below the given rows
    row_range = 10  # int(total_rows/4.00)
    begin_index = gt_row - row_range if gt_row - row_range > 0 else 0
    end_index = gt_row + row_range if gt_row + row_range < total_rows else total_rows
    for i in range(begin_index, end_index):
        for j in range(total_columns):
            edge_strength[i][j] = int(edge_strength[i][j]) * 10
    # Updating the user given value in the initial_pos_dict
    initial_pos_dict[gt_col] = gt_row
    print "Running User"
    print "Before iteration Initial pos dict: ", initial_pos_dict
    iteration = 0
    col_list = []
    for i in range(gt_col, total_columns, int(total_columns / 50)):
        col_list.append(i)
    if gt_col > 100:
        for j in range(gt_col, 0, -int(total_columns / 50)):
            col_list.append(j)
    print "col_list:", len(col_list), col_list
    for new_col2 in col_list:
        new_ridge = find_ridge_user(new_col2)
        ridges.append(new_ridge)

    optimal_ridge = convergeRidges(ridges)
    return optimal_ridge


ridge = [edge_strength.shape[0] / 2] * edge_strength.shape[1]
print (ridge)

start_time = time.time()
ridge_Bayes_Net = Bayes_Net(edge_strength)
ridge_HMM_MCMC = HMM_MCMC(edge_strength)
ridge_HMM_user = HMM_MCMC_user(edge_strength, gt_row, gt_col)

# output answer
# imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
imsave(output_filename, draw_edge(input_image, ridge_Bayes_Net, (255, 0, 0), 9))
imsave(output_filename, draw_edge(input_image, ridge_HMM_MCMC, (0, 0, 255), 7))
imsave(output_filename, draw_edge(input_image, ridge_HMM_user, (0, 255, 0), 4))
end_time = time.time()
print "Total time: ", end_time - start_time
