import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import KDTree
from sklearn import tree
from sklearn.neighbors.classification import KNeighborsClassifier
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold




data = np.genfromtxt('train .csv', delimiter=",", skip_header=1,dtype='str')
target = np.genfromtxt('train .csv', delimiter=",", skip_header=1,usecols=(80))

symbol_array=np.ndarray([80,10000],dtype='object') #儲存所有代號所代表的東西 分80項存 每項最多存10000種
array_num=np.ndarray([80],dtype='int') #記錄每想存了幾個代號
symbol_array[:,0]='NA' #全部第一項都放入NA
#print(i)
array_num[:]=1
'''
i[:]='123456789'
i[0,0]=data[0,2]
print(i)
print(data[0,2])
if(data[0,2]  in i):
    print(0)
else:
    print(1)
'''
print(target.shape[0])
#從這裡開始改data
for i in range(0,80):
    for j in range(0,target.shape[0]):
        order=0
        if data[j,i] in symbol_array[i]:
            for q in range(0,array_num[i]):
                if symbol_array[i,q] == data[j,i]:
                    order=q
                    break
        else:
            symbol_array[i,array_num[i]]=data[j,i]
            order = array_num[i]
            array_num[i]+=1
        data[j,i]=order
 #改完

        
            
            





k=data[:,0:79]

attribute = data[:,0:79]
target1 = (target/1000)
target1 = target1.astype(np.int32)
target1 = target1*1000

target2 = target1+1000
count_0=0
count_1000=0
count_2000=0
count_5000=0
count_10000=0
traintime=0
testtime=0
#target1 = target1.astype(np.str)

size=attribute.shape[0]
#print (size)
'''
print(data)
#print(target.shape[0])
print(target1)
print(target2)
'''

dtc= tree.DecisionTreeClassifier(max_depth=10)

kf2= KFold(n_splits=10, shuffle=True)
normal= []
#set 5-fold cross validation
###############################################

for train, test in kf2.split(data[:,0:79], target):
    train_time = timer()
    r=dtc.fit(data[train], target1[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = dtc.predict(data[test])
    test_time = timer()-test_time
    testtime=testtime+test_time
    traintime=traintime+train_time
    #print(pred)
    #print(target[test])
    for j in range(0,pred.shape[0]) :
        if pred[j]==target[test][j] :
            count_0=count_0+1
        if abs(pred[j]-target[test][j])<=1000:
            count_1000=count_1000+1
        if abs(pred[j]-target[test][j])<=2000:
            count_2000=count_2000+1
        if abs(pred[j]-target[test][j])<=5000:
            count_5000=count_5000+1
        if abs(pred[j]-target[test][j])<=100000:
            count_10000=count_10000+1
#    print("Train elapsed time =", train_time)
#    print("Test elapsed time =", test_time)

#    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    cnf= confusion_matrix(target[test], pred)
    #print(cnf)
#print predict target and confusion matrix
###############################################

print("|||||||||||||||||||||||||||")
print("\n decision tree (pre-pruning depth<10)")
print("train 時間" ,traintime/10)
print("test 時間" ,testtime/10)
print("完全命中=" ,count_0/target.shape[0])
print("誤差1000以內=" ,count_1000/target.shape[0])
print("誤差2000以內=" ,count_2000/target.shape[0])
print("誤差5000以內=" ,count_5000/target.shape[0])
print("誤差10000以內=" ,count_10000/target.shape[0])

count_0=0
count_1000=0
count_2000=0
count_5000=0
count_10000=0
traintime=0
testtime=0

dtc= tree.DecisionTreeClassifier(max_depth=20)

kf2= KFold(n_splits=10, shuffle=True)
normal= []
#set 5-fold cross validation
###############################################

for train, test in kf2.split(data[:,0:79], target):
    train_time = timer()
    r=dtc.fit(data[train], target1[train])
    train_time = timer()-train_time
    test_time = timer()
    pred = dtc.predict(data[test])
    test_time = timer()-test_time
    testtime=testtime+test_time
    traintime=traintime+train_time
    #print(pred)
    #print(target[test])
    for j in range(0,pred.shape[0]) :
        if pred[j]==target[test][j] :
            count_0=count_0+1
        if abs(pred[j]-target[test][j])<=1000:
            count_1000=count_1000+1
        if abs(pred[j]-target[test][j])<=2000:
            count_2000=count_2000+1
        if abs(pred[j]-target[test][j])<=5000:
            count_5000=count_5000+1
        if abs(pred[j]-target[test][j])<=100000:
            count_10000=count_10000+1
#    print("Train elapsed time =", train_time)
#    print("Test elapsed time =", test_time)


#print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
cnf= confusion_matrix(target[test], pred)
#print(cnf)
#print predict target and confusion matrix
###############################################

print("|||||||||||||||||||||||||||")
print("\n decision tree (pre-pruning depth<20)")
print("train 時間" ,traintime/10)
print("test 時間" ,testtime/10)
print("完全命中=" ,count_0/target.shape[0])
print("誤差1000以內=" ,count_1000/target.shape[0])
print("誤差2000以內=" ,count_2000/target.shape[0])
print("誤差5000以內=" ,count_5000/target.shape[0])
print("誤差10000以內=" ,count_10000/target.shape[0])


#calculate accuracy
###############################################














