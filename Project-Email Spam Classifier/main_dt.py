
from functions_dt import max_information_gain, build_tree, print_tree, test_row, print_fulltree
from Node import Node
import pandas as pd
import time as tm
import matplotlib.pyplot as plt

# Global Variables
binary_accuracy = 0
cont_accuracy = 0
binary_confusion_matrix = {}
cont_confusion_matrix = {}
depth_list = []
accuracy_list = []
accuracy_wrt_file = {}


# start = tm.clock()


#Print confusion matrix
def print_confusion_matrix(confusion_matrix, nb_type):
    # print ("\nConfusion Matrix")
    # print (confusion_matrix)
    print ("\n" + nb_type + " Confusion Matrix")
    print ("\t Model Results")
    print ("  \t T \t    N \tTotal")
    print ("T \t" + str(confusion_matrix['TP']) + "    " + str(confusion_matrix['FN']) + "\t" + str(confusion_matrix['AP']))
    print ("N \t" + str(confusion_matrix['FP']) + "    " + str(confusion_matrix['TN']) + "\t" + str(confusion_matrix['AN']))
    print ("\n")


# Prints the accuracy
def print_accuracy():
    # global binary_accuracy, cont_accuracy, binary_confusion_matrix, cont_confusion_matrix, depth_list, accuracy_list, accuracy_wrt_file
    print('\nBinary Results:')
    print('Binary Accuracy:', binary_accuracy)
    print('Confusion Matrix:')
    # print(binary_confusion_matrix)
    print_confusion_matrix(binary_confusion_matrix, "Binary")
    print('Continuous Results:')
    print('Continuous Accuracy:', cont_accuracy)
    print('Confusion Matrix:')
    # print(cont_confusion_matrix)
    print_confusion_matrix(cont_confusion_matrix, "Continuous")
    # end = tm.clock()
    # print('Time taken:', end - start)


def decision_tree_train(mode):
    training_datasets = ['panBinary_train.csv', 'panCont_train.csv']
    test_datasets = ['panBinary_test.csv', 'panCont_test.csv']

    for j in range(0, len(training_datasets)):
        training_dataset_file = training_datasets[j]
        test_dataset_file = test_datasets[j]
        depth = 0
        df = pd.read_csv(training_dataset_file, sep=",")
        train_dataset = pd.DataFrame(df)
        # print(train_dataset.head(5))
        # input('check')

        best_column, best_column_value = max_information_gain(train_dataset)
        root_node = Node(train_dataset, depth, best_column, best_column_value)
        depth_threshold = 5
        build_tree(root_node, depth_threshold)
        if mode == "train":
            if 'Binary' in training_dataset_file:
                print("\nTree built completed for Binary Dataset: \n")
                print_fulltree(root_node)
            elif 'Cont' in training_dataset_file:
                print("\nTree built completed for Continuous Dataset: \n")
                print_fulltree(root_node)

        if mode == "test":
            decision_tree_test(root_node, test_dataset_file)
    if mode == "test":
        print_accuracy()


def decision_tree_test(root_node, test_dataset_file):
    global binary_accuracy, cont_accuracy, binary_confusion_matrix, cont_confusion_matrix, depth_list, accuracy_list, accuracy_wrt_file
    test_datasets = ['panCont_test.csv', 'panBinary_test.csv']
    #for j in range(0, len(test_datasets)):
    # test
    df = pd.read_csv(test_dataset_file, sep=",")
    test_dataset = pd.DataFrame(df)
    test_dataset_count = test_dataset.shape[0]
    column_names = list(test_dataset.columns)
    class_column = column_names[0]

    # calculating depth 0 accuracy for each file
    class_0_count = test_dataset[test_dataset[class_column] == 0].shape[0]
    class_1_count = test_dataset[test_dataset[class_column] == 1].shape[0]

    # initialize TP, FN, TN and FP to 0
    TP = 0
    FN = 0

    TN = 0
    FP = 0

    predict_right_count = 0

    for index, row in test_dataset.iterrows():
        predicted_class = test_row(root_node, row)
        if predicted_class == row[class_column]:
            predict_right_count += 1
            if predicted_class == 1:
                TP += 1
            else:
                TN += 1
        else:
            if predicted_class == 1:
                FN += 1
            else:
                FP += 1

    if 'Binary' in test_dataset_file:
        # print (test_dataset_file, "If")
        # print ("Predict Right Count", predict_right_count)
        # print ("Test Dataset Count", test_dataset_count)
        # binary_accuracy = format(predict_right_count/test_dataset_count * 100, '.2f')
        binary_accuracy = (100.0 * predict_right_count)/test_dataset_count
        # print ("Binary Accuracy", binary_accuracy)
        binary_confusion_matrix['AP'] = class_1_count
        binary_confusion_matrix['AN'] = class_0_count
        binary_confusion_matrix['TP'] = TP
        binary_confusion_matrix['FN'] = FN
        binary_confusion_matrix['TN'] = TN
        binary_confusion_matrix['FP'] = FP
    else:
        # print (test_dataset_file, "Else")
        # print ("Predict Right Count", predict_right_count)
        # print ("Test Dataset Count", test_dataset_count)
        # cont_accuracy = format(predict_right_count/test_dataset_count * 100, '.2f')
        cont_accuracy = (100.0 * predict_right_count)/test_dataset_count
        # print ("Continuous Accuracy", cont_accuracy)
        cont_confusion_matrix['AP'] = class_1_count
        cont_confusion_matrix['AN'] = class_0_count
        cont_confusion_matrix['TP'] = TP
        cont_confusion_matrix['FN'] = FN
        cont_confusion_matrix['TN'] = TN
        cont_confusion_matrix['FP'] = FP
