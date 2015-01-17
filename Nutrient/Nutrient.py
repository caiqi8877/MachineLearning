import matplotlib.pyplot as plt
import random
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl
from scipy import linalg


### read file,
# split every line by "^"
# output: [ table: appended data ]
def read_file():
    table = []
    food_names = []
    for input_file in ["dataCereal-grains-pasta.txt","dataFats-oils.txt","dataFinfish-shellfish.txt","dataVegetables.txt"]:
        file = open(input_file)
        lines = file.readlines()
        foodInfo = []
        for line in lines:
            end = line.find("\n")
            foodInfo = line[:end - 1].split("^")
            table.append(foodInfo[1:end])
            food_names.append(foodInfo[0])
    return table,food_names
#########################################################
### normalize data
# 
# 
def normalize(table):
    for index in range(0,len(table[0])):
        max_num = -100.0
        min_num = 100.0
        for i in range(0,len(table)):
            table[i][index] = float(table[i][index])
            temp = table[i][index]
            if temp > max_num:
                max_num = temp
            if temp < min_num:
                min_num = temp
        for i in range(0,len(table)):
            if (max_num - min_num) != 0:
                table[i][index] = (table[i][index] - min_num) / (max_num - min_num)
    return table
######################################################### 
### centralized data
# 
# 
def centralized(table):
    #
    #
    #
    for index in range(0,len(table[0])):
        sum_num = 0.0
       
        for i in range(0,len(table)):
            sum_num = table[i][index] + sum_num
        
        mean = sum_num / len(table)
        for i in range(0,len(table)):
            table[i][index] = table[i][index] - mean

    return table
######################################################### 
### 
#   plot the PCA figure in the first two components
# 
def plot(W):
    x1,x2,x3,x4,y1,y2,y3,y4 = [],[],[],[],[],[],[],[]
    for index in range(0,182):
        x1.append(W[index][0])
        y1.append(W[index][1])
    for index in range(182,401):
        x2.append(W[index][0])
        y2.append(W[index][1])
    for index in range(401,668):
        x3.append(W[index][0])
        y3.append(W[index][1])
    for index in range(668,1496):
        x4.append(W[index][0])
        y4.append(W[index][1])
    plt.plot(x1,y1,'ro')
    plt.plot(x2,y2,'bo')
    plt.plot(x3,y3,'yo')
    plt.plot(x4,y4,'go')
    plt.legend(['Cereal-grains-pasta','Fats-oils','Finfish-shellfish','Vegetables'], loc='lower right')
    plt.show();
######################################################### 
### 
#   plot the kmenas figure in the first two components
# 
def kmeans(c_table):
    reduced_data = PCA(n_components=2).fit_transform(c_table)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    kmeans.fit(reduced_data)
    h = .02
    x_min, x_max = reduced_data[:, 0].min() , reduced_data[:, 0].max() 
    y_min, y_max = reduced_data[:, 1].min() , reduced_data[:, 1].max() 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')
    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    pl.xticks(())
    pl.yticks(())
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.show()
######################################################### 
### compute labels accuracy
#   
# 
def randIndex(truth, predicted):
    """
    The function is to measure similarity between two label assignments
    truth: ground truth labels for the dataset (1 x 1496)
    predicted: predicted labels (1 x 1496)
    """
    if len(truth) != len(predicted):
        print "different sizes of the label assignments"
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel-1):
        for j in range(i+1,sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same +=1
            count += 1
    return (agree_same+disagree_same)/float(count)

######################################################### 
### 
#   generate 3 labels for the question 5, truth, permutation of truth and kmenas label
# 
def labels_generator(n_table):
    reduced_data = PCA(n_components=2).fit_transform(n_table)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    kmeans.fit(reduced_data)

    #estimator.labels_
    labels = [0,1,2,3]
    truth = [labels[0]] * 182 + [labels[1]] * 219 + [labels[2]] * 267 + [labels[3]] * 828
    random.shuffle(labels)
    ground_permutate_labels = [labels[0]] * 182 + [labels[1]] * 219 + [labels[2]] * 267 + [labels[3]] * 828
    return truth,ground_permutate_labels,kmeans.labels_

######################################################### 
### question 6
#   set n_init to 1 and run 20 times to view the kmenas objective
# 
def kmeans_objective(n_table,truth):
    reduced_data = PCA(n_components=2).fit_transform(n_table)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=1)
    kmeans.fit(reduced_data)
    if(kmeans.inertia_ > 130.237819549):
        print randIndex(truth,kmeans.labels_)
    return kmeans.inertia_
######################################################### 
### 
#   view the dendogram of the data random choose 30 from each cluster
# 
def view_dendogram(n_table):
    fig = pl.figure() 

    table1 = random.sample(n_table[0:182],30)
    table2 = random.sample(n_table[182:401],30)
    table3 = random.sample(n_table[401:668],30)
    table4 = random.sample(n_table[668:1496],30)
    data = np.concatenate((table1, table2, table3, table4), axis=0)
 
    datalable = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30
    
    hClsMat = sch.linkage(data, method='complete') # Complete clustering
    sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45)
    pl.show()
######################################################### 
### 
#   
# 
def fcluster(n_table,truth,W):
    hClsMat = sch.linkage(n_table, method='complete')
    resultingClusters = sch.fcluster(hClsMat,t= 3.8, criterion = 'distance')
    print randIndex(truth,resultingClusters)

    x1,x2,x3,x4,y1,y2,y3,y4 = [],[],[],[],[],[],[],[]
    for i in range(len(resultingClusters)):
        if resultingClusters[i] == 1 :
            x1.append(W[i][0])
            y1.append(W[i][1])
        elif resultingClusters[i] == 2 :
            x2.append(W[i][0])
            y2.append(W[i][1])
        elif resultingClusters[i] == 3 :
            x3.append(W[i][0])
            y3.append(W[i][1])
        elif resultingClusters[i] == 4 :
            x4.append(W[i][0])
            y4.append(W[i][1])
    plt.plot(x1,y1,'ro')
    plt.plot(x2,y2,'bo')
    plt.plot(x3,y3,'yo')
    plt.plot(x4,y4,'go')
    plt.show();
######################################################### 
### 
#   set n_clusters to 5,10,25,50,75 and display the largest cluster size and related items
# 
def largest_cluster(cereal_table,food_names):
    estimator = KMeans(init='k-means++', n_clusters=75, n_init=40)
    reduced_data = PCA(n_components=2).fit_transform(cereal_table)
    estimator.fit(reduced_data)
    l = estimator.labels_.tolist()
    max_num = 0
    maxitem = None
    for x in set(l):
        count =  l.count(x)
        if count > max_num:
            max_num = count
            maxitem = x
    index = []
    for i in range(len(l)):
        if l[i] == maxitem:
            index.append(i)
    if len(index) >= 10:
        index = random.sample(index,10)
    ten_items = []
    for i in range(len(index)):
        ten_items.append(food_names[index[i]])

    return max_num,ten_items
######################################################### 
### 
#   view the proportion of variance explained by the first 10 principal components.
# 
def variance_plot(s):

    square_sum = 0.0
    for i in range(len(s)):
        square_sum = square_sum + s[i] * s[i]
    
    other = 0.0
    t= []
    for i in range(10):
        stemp = (s[i] * s[i]) / square_sum
        other = other + stemp
        t.append(stemp);

    labels = '1','2','3','4','5','6','7','8','9','10','other components variance'
    fracs = [t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],1-other]
    explode=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.05)
    plt.pie(fracs, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)
    plt.show()

######################################################### 

#   reconstruct data from first 5 to 149 component and view the reconstruction error
# 
def plot_ab(n_table):
    
    k = [5,10,25,50,75,100,149]
    table = np.array(n_table)
    mean = sum(table)/len(table)
    n_table = n_table - mean
    U, s, Vh = linalg.svd(n_table)
    W = np.dot(n_table,Vh.T)
    error = []
    for i in range(len(k)):
        temp_table = np.dot(np.array(W[:,0:k[i]]), np.array(Vh[:k[i],:]))
        temp_table = temp_table + mean
        temp = n_table - temp_table
        temp = temp * temp
        error_sum = sum(sum(temp))
        error.append(error_sum)
    E = []
    for index in k:
        temp = 0
        for i in range(index,150):
            temp = s[i] * s[i] + temp
        E.append(temp) 
    plt.plot(k,error)
    plt.plot(k,E,'r--')
    plt.legend(['Reconstruct error','Ek'], loc='upper right')
    plt.show()

def find_weight():
    file = open("dataDescriptions.txt")
    line = file.readline()
    end = line.find("\n")
    nutrients = line[:end - 1].split("^")
    nutrients = nutrients[1:]
    # print nutrients
    # a = list(abs(Vh[0]))
    a = list(abs(Vh[1]))
    lis = []
    num = []
    for k in range(5):
        lis.append(a.index(max(a)))
        a[a.index(max(a))] = -100
    print lis
    for i in lis:
        print Vh[1][i]," ",nutrients[i]
#########################################################        
def main():
    
    table,food_names = read_file()
    n_table = normalize(table)
    c_table = centralized(n_table)
    
    U, s, Vh = linalg.svd(c_table)
    W = np.dot(c_table,Vh.T)
    
    plot(W);
    find_weight(Vh);
    
    kmeans(n_table)
    
    truth,ground_permutate_labels,kmeans_predict = labels_generator(n_table)
    
    obj = kmeans_objective(n_table,truth)
    
    view_dendogram(n_table)
     
    fcluster(n_table,truth,W)
    for ti in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8]:
        hClsMat = sch.linkage(n_table, method='complete')
        resultingClusters = sch.fcluster(hClsMat,t= ti, criterion = 'distance')
        print ti,randIndex(truth,resultingClusters)
    #9
    cereal_table = n_table[0:182]
    max_num,ten_items = largest_cluster(cereal_table,food_names)
    print max_num
    for i in range(len(ten_items)):
        print ten_items[i]
    
    variance_plot(s)
    
    plot_ab(n_table)
    

if __name__ == "__main__":
    main()
