dot -Tpdf iris.dot -o iris.pdf

import matplotlib.pyplot as plt
import random
import os
from sklearn import tree
from sklearn import ensemble

    ### global variable
    #  feature_number : the number of features in feature.txt
    #  continus_number: in 12 features, there are 4 features are numerical like age, captain gain
    #  sampleNum: refer to the change from 1 to 30(50) of the size fo tree, the min sample leaf
feature_number = 12
continus_number = 4
sampleNum = 2

    ### read file,
    # para  input_file: file name
    # output:    table: two dimention array represent the input data
def read_file(input_file):
    file = open(input_file)
    lines = file.readlines()
    table = []
    personInfo = []
    for line in lines:
        end = line.find("\n")
        personInfo = line[:end].split(" ")
        table.append(personInfo)
    return table
    ### read features.txt,
    # input:     12 kinds of features
    # output:    length of 12 two dimension array

def testTree(train,validation):
    colLength = len(train[0])

    # X is training input
    # Y is real result of training set
    # VX is validation input
    # Y is real result of validation set 

    X,Y,VX,VY = [],[],[],[]
    for row in range(0,len(train)):
        X.append(train[row][1:colLength])
        Y.append(train[row][0:1])
    for row in range(0,len(validation)):
        VX.append(validation[row][1:colLength])
        VY.append(validation[row][0:1])
 
    t_accuracy = [0.0] * sampleNum
    v_accuracy = [0.0] * sampleNum
  
    clf = tree.DecisionTreeClassifier()
    maxDep,maxnum = 0,0;
    for j in range(0,sampleNum):

        ##  max_depth range from 1 to 30
        clf.max_depth = j + 1

        ##  min_samples_leaf range from 1 to 50
        # clf.min_samples_leaf = j + 1
        ### Using RandomForest
        # clf = ensemble.RandomForestClassifier(n_estimators = j + 1)
        # clf = ensemble.RandomForestClassifier(n_estimators = j + 1, max_depth = 10,  min_samples_leaf = 44)  
        clf = clf.fit(X, Y)

        #predict validation result
        PY = clf.predict(VX)

        #predict validation result
        TY = clf.predict(X)
        for i in range(0,len(PY)):

        ## if predict right, validation plus 1
            if PY[i] == VY[i] :
                v_accuracy[j] = v_accuracy[j] + 1.0

        ## if predict right, test plus 1
            if TY[i] == Y[i] :
                print TY[i]
                print Y[i]
                t_accuracy[j] = t_accuracy[j] + 1.0

        ## divide the length get the accuracy presentage
        v_accuracy[j] = v_accuracy[j]/len(PY)

        ## used to find the highest validation accuracy and relate data
        if(v_accuracy[j] > maxDep):
            maxDep,maxnum = v_accuracy[j],j+1
        ## compute train accuracy
        t_accuracy[j] = t_accuracy[j]/len(PY)
    print maxnum
    return v_accuracy,t_accuracy

def plot_pic(v_acc,t_acc):

    ### plot the train accuracy and validation accuracy
    ### they are coloured blue and red respectively
    x = range(1,sampleNum + 1)
    plt.gca().set_color_cycle(['black', 'red'])
    plt.plot(x,t_acc)
    plt.plot(x,v_acc)
    plt.legend(['train accuracy','validation accuracy'], loc='lower right')
    plt.show();
    print v_acc

def visualize_decision_tree(train):

    ### using the best configuration max_depth = 10 and min_sample_leaf = 44
    #   and graphviz the decision tree , limit the depth of the decision tree to 3
    colLength = len(train[0])
    X,Y,VX,VY = [],[],[],[]
    for row in range(0,len(train)):
        X.append(train[row][:colLength-1])
        Y.append(train[row][colLength-1])
    clf = tree.DecisionTreeClassifier()
    clf.max_depth = 10
    clf.min_samples_leaf = 44
    clf = clf.fit(X, Y)
    export_file = tree.export_graphviz(clf,out_file='iris.dot',max_depth=3)
    
def main():
    
    input_file = "train.txt"
    train = read_file(input_file)

    input_file1 = "test.txt"
    test = read_file(input_file1)

     
    v_acc,t_acc = testTree(train,test)
    plot_pic(v_acc,t_acc)
    
    visualize_decision_tree(train)
    
    test your data
    input_file1 = "adult_test.txt"
    test_table = read_file(input_file1)
    handle_missing(test_table)
    new_test_table = new_feature(test_table,feature_table)
    v_acc,t_acc = testTree(new_feature_table,new_test_table)
    plot_pic(v_acc,t_acc)

    
if __name__ == "__main__":
    main()