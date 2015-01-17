import matplotlib.pyplot as plt
training_number = 4000
threshold = 0.0
learning_rate = 1.0
most_words_wanted = 15 
dictionary_words_num = 30 #global variables

def split_train_validation_data():
    f_train = open("spam_for_train.txt",'w')
    f_validation = open("spam_for_validation.txt",'w')
    file = open("spam_train.txt")
    lines = file.readlines()
    line_count = 0
    for line in lines:
        if line_count < training_number:
            f_train.write(str(line))
        else :
            f_validation.write(str(line))
        line_count += 1

def creat_dictionary():
    file = open("spam_for_train.txt")
    dic = {}
    dictionary = {}
    index = 0
    lines = file.readlines()
    for line in lines:
        line_email = line[1:]
        split_words = set(line_email.split(' '))  #remove duplicate words in an email
        for word in split_words:
            if word in dic:
                dic[word] = dic[word] + 1
            else: 
                dic[word] = 1 #if word not in the dictionary, create it
    dictionary_list = []
    dicts = []
    for key in dic:
        if dic[key] >= dictionary_words_num:  #first 30 most frequent appearing words
            dicts.append(key) 
            dictionary[key] = index  # put it in the dictionary
            dictionary_list.append(key)  # also put it in the list for question 5 mainy
            index += 1
    return (dictionary,dictionary_list)

def creat_table(dictionary):
    table = []
    desired_result_list = []
    file = open("spam_for_train.txt")
    lines = file.readlines()
    for line in lines:
        desired_result = line[0:1]   # retrieve the first elem, 0/1 , whether it is the spam email
        desired_result_list.append(float(desired_result))  
        line_row = line[1:].split(' ')
        email = [0.0] * len(dictionary)  #set feature to be [0]*2376 first
        for word in line_row:
            if word in dictionary:
                email[dictionary[word]] = 1.0 #if word in dictionary, set the corresponding word to be 1
        table.append(email)  
    for i in range(0,len(table)):
         table[i].insert(0,desired_result_list[i]) #assert the 0/1 elem into the table as the desired result the the training set
    return table

def creat_validation_table(dictionary): #same as creat_table, only different is using the last 1000 emal
    table = []
    desired_result_list = []
    file = open("spam_for_validation.txt")
    lines = file.readlines()
    for line in lines:
        desired_result = line[0:1]
        desired_result_list.append(float(desired_result))
        line_row = line[1:].split(' ')
        email = [0.0] * len(dictionary)
        for word in line_row:
            if word in dictionary:
                email[dictionary[word]] = 1.0
        table.append(email)
    for i in range(0,len(table)):
         table[i].insert(0,desired_result_list[i])
    return table

def dot_product(v1,v2): # return dot_product of two array
    result = 0.0
    for i in range(len(v1)):
        result += v1[i]* v2[i]
    return result

def perceptron_train(data): 
    weight_vector = [0.0] * ( len(data[0])-1 )
    iter = 0 ; # number of iterations
    updated = 0 # number of errors
    while True:
        final_loop = 1 # if final_loop doesn't change, means no error in this pass, thus the loop breaks,
        iter += 1
        print iter # using for debug
        for i in range(0,len(data)):
            temp_result = dot_product(weight_vector, data[i][1:])
            if (temp_result > threshold): #if dot_product temp_result is above 0,set it to 1
                temp_result = 1.0
            else:                         #if dot_product temp_result is below 0,set it to 0
                temp_result = 0.0
            if temp_result > data[i][0]:  #if dot_product is too high,it minus the weight vector 
                updated = updated + 1
                for j in range(len(weight_vector)):
                    weight_vector[j] -= learning_rate*data[i][j+1]
                final_loop = 0
            elif temp_result < data[i][0]: #if dot_product is too low,it add the weight vector
                updated = updated + 1
                for j in range(0,len(weight_vector)):
                    weight_vector[j] += learning_rate*data[i][j+1]
                final_loop = 0
        if (final_loop == 1) : #if no error in this pass, then break the loop
            break
    return (weight_vector,updated,iter)

def weighted_perceptron_train(data):
    weight_matrix = []
    weight_vector = [0.0] * ( len(data[0])-1 )
    iter = 0 ;
    updated = 0
    while True:
        final_loop = 1
        iter += 1
        print iter
        for i in range(0,len(data)):
            temp_result = dot_product(weight_vector, data[i][1:])
            temp_vector = weight_vector #save the weight vector into a temp vector
            weight_matrix.append([]+temp_vector) #every time,whether the vector updated or not, put it in a two-dimension array 
            if (temp_result > threshold):
                temp_result = 1.0
            else:
                temp_result = 0.0
            if temp_result > data[i][0]:
                updated = updated + 1
                for j in range(len(weight_vector)):
                    weight_vector[j] -= learning_rate*data[i][j+1]
                final_loop = 0
            elif temp_result < data[i][0]:
                updated = updated + 1
                for j in range(0,len(weight_vector)):
                    weight_vector[j] += learning_rate*data[i][j+1]
                final_loop = 0
        if (final_loop == 1) :
            break
            
    vector = []
    for j in range(0,len(weight_matrix[0])):#the weight_matrix store every temp weight vector
        s = 0
        for i in range(0,len(weight_matrix)):
            s += weight_matrix[i][j]
        vector.append(s/len(weight_matrix)) # get the average of every vectors
    return (vector,updated,iter)

def perceptron_test(w,data):
    print len(data)
    correct = 0.0
    for i in range(0,len(data)):
        temp_result = dot_product(data[i][1:],w)
        if temp_result > threshold:
            output = 1.0
        else:
            output = 0.0
        if ( output == data[i][0]): #check if it is right
            correct += 1.0
    return 1.0 - correct/(float)(len(data)) #compute the error presentage

def most_weight_words(words,a):
    most_p_words = []
    most_n_words = []
    for i in range(0,most_words_wanted):
        most_p_words.append(words[a.index(max(a))]) # find the most frequent words and save it in most_p_words
        most_n_words.append(words[a.index(min(a))]) 
        words.remove(words[a.index(max(a))]) # delete the most frequent word so next time the loop will find the 2nd most frequent word
        a.remove(max(a))
        words.remove(words[a.index(min(a))])
        a.remove(min(a))
    print most_p_words
    print most_n_words

def train_number_with_iter(test_list,table):
    iter_list = []
    for train_number in test_list:
        data = table[0:train_number]
        (weight_vector,k,iter) = perceptron_train(data)
        iter_list.append(iter)
    print iter_list

def perceptron_train_number_with_error(test_list,table,v_data):
    error_list = []
    for train_number in test_list:
        data = table[0:train_number]
        (weight_vector,k,iter) = perceptron_train(data)
        error = perceptron_test(weight_vector,v_data)
        error_list.append(error)
    print error_list

def weighted_perceptron_train_number_with_error(test_list,table,v_data):
    error_list = []
    for train_number in test_list:
        data = table[0:train_number]
        (weight_vector,k,iter) = weighted_perceptron_train(data)
        error = perceptron_test(weight_vector,table)
        error_list.append(error)
    print error_list

def max_iter_perceptron_train(data,max_iter_num): #just add one argument to the perceptron_train function
    weight_vector = [0.0] * ( len(data[0])-1 )
    iter = 0 ;
    updated = 0
    line_count = 0
    while True:
        final_loop = 1
        iter += 1
        print iter
        for i in range(0,len(data)):
            temp_result = dot_product(weight_vector, data[i][1:])
            if (temp_result > threshold):
                temp_result = 1.0
            else:
                temp_result = 0.0
            if temp_result > data[i][0]:
                updated = updated + 1
                for j in range(len(weight_vector)):
                    weight_vector[j] -= learning_rate*data[i][j+1]
                final_loop = 0
            elif temp_result < data[i][0]:
                updated = updated + 1
                for j in range(0,len(weight_vector)):
                    weight_vector[j] += learning_rate*data[i][j+1]
                final_loop = 0
        line_count += 1
        if line_count == max_iter_num:
            break
        if (final_loop == 1) :
            break
    return (weight_vector,updated,iter)

def max_iter_weighted_perceptron_train(data,max_iter_num):  #just add one argument to the weighted perceptron_train function
    weight_matrix = []
    weight_vector = [0.0] * ( len(data[0])-1 )
    iter = 0 ;
    updated = 0
    line_count = 0
    while True:
        final_loop = 1
        iter += 1
        print iter
        for i in range(0,len(data)):
            temp_result = dot_product(weight_vector, data[i][1:])
            temp_vector = weight_vector
            weight_matrix.append([]+temp_vector)
            if (temp_result > threshold):
                temp_result = 1.0
            else:
                temp_result = 0.0
            if temp_result > data[i][0]:
                updated = updated + 1
                for j in range(len(weight_vector)):
                    weight_vector[j] -= learning_rate*data[i][j+1]
                final_loop = 0
            elif temp_result < data[i][0]:
                updated = updated + 1
                for j in range(0,len(weight_vector)):
                    weight_vector[j] += learning_rate*data[i][j+1]
                final_loop = 0
        line_count += 1
        if line_count == max_iter_num:
            break
        if (final_loop == 1) :
            break
            
    vector = []
    for j in range(0,len(weight_matrix[0])):
        s = 0
        for i in range(0,len(weight_matrix)):
            s += weight_matrix[i][j]
        vector.append(s/len(weight_matrix))
    return (vector,updated,iter)

def main():
#-------------------------------------- question 1
    # split_train_validation_data()
#-------------------------------------- question 2
    (dictionary,dictionary_list) = creat_dictionary()
    table = creat_table(dictionary)
#-------------------------------------- question 3
    (weight_vector,k,iter) = perceptron_train(table)
    print "times: ",k
    
    v_data = creat_validation_table(dictionary)
    test_error = perceptron_test(weight_vector,v_data)
    print test_error
#-------------------------------------- question 5
    # most_weight_words(dictionary_list,weight_vector)
#-------------------------------------- question 6
    # (weight_vector,k,iter) = weighted_perceptron_train(table)
#-------------------------------------- question 7
    #test_list = [100,200,400,800,2000,4000]
    # error_list = perceptron_train_number_with_error(test_list,table,v_data)
    # error_list = weighted_perceptron_train_number_with_error(test_list,table,v_data)
  #-------------------------------------- question 8
    # iter_list = train_number_with_iter(test_list,table)
#-------------------------------------- question 9
    # (weight_vector,k,iter) = max_iter_perceptron_train(table,max_iter_num)
    # (weight_vector,k,iter) = max_iter_weighted_perceptron_train(table,max_iter_num)
#-------------------------------------- question 10
    #change global variable training_number to 5000 and dictionary_words_num to 33
    #set iter to be 5 times which in the validation test is the lowest 
    # (weight_vector,k,iter) = max_iter_weighted_perceptron_train(table,5)
    #in the function creat_validation_table(),change read file to spam_test.txt
    # v_data = creat_validation_table(dictionary)
    # test_error = perceptron_test(weight_vector,v_data)
    # print test_error

if __name__ == "__main__":
    main()