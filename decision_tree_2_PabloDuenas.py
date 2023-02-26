#-------------------------------------------------------------------------
# AUTHOR: Pablo Duenas
# FILENAME: decision_tree_2_PabloDuenas
# SPECIFICATION: Using decision trees to calculate accuracy. Read 3 different data training files of different number of instances to see how it affects model's accuracy.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
count = 0

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    for row in dbTraining:
        if row[0] == "Young":
            d1 = 1
        elif row[0] == "Prepresbyopic":
            d1 = 2
        elif row[0] == "Presbyopic":
            d1 = 3
        
        if row[1] == "Myope":
            d2 = 1
        elif row[1] == "Hypermetrope":
            d2 = 2
        
        if row[2] == "Yes":
            d3 = 1
        elif row[2] == "No":
            d3 = 2
        
        if row[3] == "Normal":
            d4 = 1
        elif row[3] == "Reduced":
            d4 = 2
        
        X.append([d1, d2, d3, d4])

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for row in dbTraining:
        Y.append(1) if row[4] == "Yes" else Y.append(2)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        dbTest = []
        with open('contact_lens_test.csv', 'r') as testcsv:
            testReader = csv.reader(testcsv)
            for i, row in enumerate(testReader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            testData = []

            if data[0] == "Young":
                d1 = 1
            elif data[0] == "Prepresbyopic":
                d1 = 2
            elif data[0] == "Presbyopic":
                d1 = 3
        
            if data[1] == "Myope":
                d2 = 1
            elif data[1] == "Hypermetrope":
                d2 = 2
        
            if data[2] == "Yes":
                d3 = 1
            elif data[2] == "No":
                d3 = 2
        
            if data[3] == "Normal":
                d4 = 1
            elif data[3] == "Reduced":
                d4 = 2
        
            testData = [d1, d2, d3, d4]
            class_actual = 1 if data[4] == "Yes" else 2
            class_predicted = clf.predict([testData])[0]

            if class_predicted == 1 and class_actual == 1:
                TP += 1
            elif class_predicted == 2 and class_actual == 2:
                TN += 1
            elif class_predicted == 1 and class_actual == 2:
                FP += 1
            elif class_predicted == 2 and class_actual == 1:
                FN += 1


    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

    modelAccuracy = (TP+TN)/(TP+TN+FP+FN)

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print("Final accuracy when training on {}: {}".format(ds, modelAccuracy))




