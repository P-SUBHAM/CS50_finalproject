
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('DJIA.csv')
yy = dataset.iloc[:, 4].values
xy = np.zeros(3100)

# Normalization
for i in range(len(xy)):
    xy[i] = (yy[i] - min(yy)) / (max(yy) - min(yy))

# Splitting the dataset into the Training set and Test set
x_train = xy[:int(-(10/31)*len(xy))]
x_test = xy[int(-(10/31)*len(xy)):]

del yy

# Creating the sliding matrix of training set and test set
#initializations for training set
learning_rate = 0.0091
k = 0
t = 100-1
s_train = 2000
m_train = np.zeros(s_train) #mean
v_train = np.zeros(s_train) #variance
xx_train = np.zeros(shape=(s_train,6))

#sliding window for training set
for i in range(s_train):
    inp = [x_train[t-27], x_train[t-20], x_train[t-13], x_train[t-6], x_train[t]]
    inp = np.asarray(inp)
    m_train[i] = inp.mean();
    v_train[i] = ( sum(inp**2)/5 ) - ( m_train[i]**2 ) 
    v_train[i] = np.sqrt(v_train[i])
    for j in [t-27, t-20, t-13, t-6, t, t+1]:
        xx_train[i,k] = x_train[j]
        k = k+1
    t = t+1
    k=0
    
#inititializations for test set
k = 0
t = 50-1
s_test = 900
m_test = np.zeros(s_test) #mean
v_test = np.zeros(s_test) #variance
xx_test = np.zeros(shape=(s_test,6))

#sliding window for testing set
for i in range(s_test):
    inp = [x_train[t-27], x_train[t-20], x_train[t-13], x_train[t-6], x_train[t]]
    inp = np.asarray(inp)
    m_test[i] = inp.mean();
    v_test[i] = ( sum(inp**2)/5 ) - ( m_test[i]**2 ) 
    v_test[i] = np.sqrt(v_test[i])
    for j in [t-27, t-20, t-13, t-6, t, t+1]:
        xx_test[i,k] = x_test[j]
        k = k+1
    t = t+1
    k=0
    
import jaya

print("JAYA algorithm initiated---\n")
w,mse_train,y_pred_train = jaya.jaya_train(s_train, m_train, v_train, xx_train)

print("\n\nOptimized Weight Found: ", w,"\n\n")

print("Testing predictions with optimized weights---\n")
mse_test,y_pred_test = jaya.jaya_test(s_test, m_test, v_test, xx_test, w)

# Visualising the Training set results
plt.figure()
plt.plot(mse_train)
plt.title('MSE (Training set)')
plt.xlabel('No of Iterations')
plt.ylabel('MSE')
plt.show()

s_train = 700
train_iter = np.zeros((s_train,1))
for i in range(s_train):
    train_iter[i] = i

plt.figure()
plt.axis([0,s_train, -3*max(xx_train[:s_train,5]) , 3*max(xx_train[:s_train,5])])

plt.plot(train_iter, xx_train[:s_train,5], color = 'red')
#plt.hold()
plt.plot(train_iter, y_pred_train[:s_train], color = 'blue')
plt.title('Training set')
plt.xlabel('No of Iterations')
plt.ylabel('Stock Price')
red_patch = mpatches.Patch(color='red', label='Actual Data')
blue_patch = mpatches.Patch(color='blue', label='Predicted Data')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

# Visualising the Test set results
plt.figure()
plt.plot(mse_test)
plt.title('MSE (Testing set)')
plt.xlabel('No of Iterations')
plt.ylabel('MSE')
plt.show()

s_test = 700
test_iter = np.zeros((s_test,1))
for i in range(s_test):
    test_iter[i] = i

plt.figure()
plt.axis([0,s_test, -3*max(xx_test[:s_test,5]) , 3*max(xx_test[:s_test,5])])

plt.plot(test_iter, xx_test[:s_test,5], color = 'red')
#plt.hold()
plt.plot(test_iter, y_pred_test[:s_test], color = 'blue')
plt.title('Testing set')
plt.xlabel('No of Iterations')
plt.ylabel('Stock Price')
red_patch = mpatches.Patch(color='red', label='Actual Data')
blue_patch = mpatches.Patch(color='blue', label='Predicted Data')
plt.legend(handles=[red_patch, blue_patch])

plt.show()

sum1 = 0
for i in range(s_test):
    sum1 += abs(y_pred_test[i] - xx_test[i,5]) 
accuracy = sum1/s_test
print("\n\nAccuracy : ",100 - accuracy)