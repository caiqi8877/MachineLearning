from svm import *
from svmutil import *
import matplotlib.pyplot as plt

_lambda = pow(2,-5);
IterTimes = 20;
k = 5
def scale_train_data():
    file_num = open("mnist_train.txt")
    lines = file_num.readlines()
    numbers = []
    table = []
    for line in lines:
    	numbers = line.split(',');
        numbers = map(float,numbers);
        for i in range(1,len(numbers)):
            numbers[i] = 2*numbers[i]/255 - 1 ;
        table.append(numbers);
    return table

def scale_test_data():
    file_num = open("mnist_test.txt")
    lines = file_num.readlines()
    numbers = []
    table = []
    for line in lines:
        numbers = line.split(',');
        numbers = map(float,numbers);
        for i in range(1,len(numbers)):
            numbers[i] = 2*numbers[i]/255 - 1 ;
        table.append(numbers);
    return table

def pegasos_svm_train(table,_lambda):
    t = 0.0;
    w_length = len(table[0]) - 1
    w = [[0.0 for i in range(w_length)] for j in range(10)]
    for n in range(0,10):
        yi = [0.0] * len(table)
        for index in range(0,len(table)):
            if(table[index][0] == n ):
                yi[index] = 1.0
            else:
                yi[index] = (-1.0)
        for iter in range (0,IterTimes):
            print (iter+1)
            for j in range (0,len(table)):
                t = t + 1;
                xi = table[j][1:];
                # yi = table[j][0];
                eta = 1.0 / (t * _lambda);
                if( yi[j] * dot_product(w[n],xi) < 1.0):
                    for i in range(0, len(w[n])):
                        w[n][i] = ( 1.0 - eta * _lambda) * w[n][i] + eta * xi[i] * yi[j];
                else:
                    for i in range(0, len(w)):
                        w[n][i] = ( 1.0 - eta * _lambda) * w[n][i];
    return w

# return dot_product of two array

def dot_product(v1,v2): 
    result = 0.0
    for i in range(len(v1)):
        result += v1[i]* v2[i]
    return result

def svm_test(data,w):
    error = 0.0;
    for i in range(1,len(data)):
        predict = -100;
        for j in range(0,10):
            temp = dot_product(data[i][1:],w[j])
            if( temp > predict):
                predict = temp
                predictNum = j
        if(data[i][0] != predictNum):
            error = error + 1;
            
    return error/(float(len(data)))

    

def main():

    table = scale_train_data()

    w = pegasos_svm_train(table,_lambda)
    error = svm_test(table,w)
    print error

   
    fold_list = range(0,len(table)+1,len(table)/k)
    temp_table = []
    for i in range(0,k):
        temp_table.append(table[fold_list[i]:fold_list[i+1]])
    total_error = 0.0
    for fold_time in range(0,k):
        print ("fold_time",fold_time)
        data = []
        vdata = []
        for i in range(0,k):
            if(i != fold_time):
                data += temp_table[i]
            else:
                vdata = temp_table[i]

        w = pegasos_svm_train(data,_lambda)
        error = svm_test(vdata,w)
        total_error = total_error + error
    print total_error/k

    w = pegasos_svm_train(table,_lambda)
    ttable = scale_test_data()
    error = svm_test(ttable,w)
    print error
  
    ttable = scale_test_data()
    y,x,yt,xt = [],[],[],[]
    for i in range(0,len(table)):
        y.append(table[i][0])
        x.append(table[i][1:])
    for i in range(0,len(ttable)):
        yt.append(ttable[i][0])
        xt.append(ttable[i][1:])
    m = svm_train(y, x)
    p_labels, p_acc, p_vals = svm_predict(yt, xt, m)

if __name__ == "__main__":
    main()