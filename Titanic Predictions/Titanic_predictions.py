#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:47:21 2020

@author: swapnillagashe
"""

#Titanic Predictions model
#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
os.chdir('/Users/DATA/Coding /Kaggle /Titanic Predictions/Data')
#import data
data_original=pd.read_csv('train.csv')
data=data_original.copy()
data.info()
data.isnull().sum()

data['Embarked'].fillna(value='S',inplace=True)
data['family']=data['SibSp']+data['Parch']+1
data['Sex'] = data['Sex'].replace(['female','male'],[0,1])
data['Embarked'] = data['Embarked'].replace(['S','Q','C'],[1,2,3])
data['Title']=data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

data['Title'] = data['Title'].map(title_mapping)
#Assign median age of particular group to missing values
data["Age"].fillna(data.groupby("Title")["Age"].transform("median"), inplace=True)
# fill missing Fare with median fare for each Pclass
data["Fare"].fillna(data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#cabin
data['Cabin'] = data['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8} #we are scaling here
data['Cabin'] = data['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass
data["Cabin"].fillna(data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


data = data.drop(['PassengerId','Survived','Name', 'SibSp','Parch','Ticket'], axis=1)
input_data=data.copy()
input_data.isnull().sum() #check if any zero values are present in the data



labels_data=pd.DataFrame()
labels_data['Yes']=data_original['Survived']
labels_data['No']= 1-data_original['Survived']
#scaling the data
#sc=StandardScaler()
#input_data[['Age','Fare']] = sc.fit_transform(input_data[['Age', 'Fare']])

input_data=input_data.to_numpy()
labels_data=labels_data.to_numpy()







trainX, testX, trainY, testY = train_test_split(input_data, labels_data, test_size=0.33, random_state=42)

#parameters
numFeatures=trainX.shape[1]
numLabels = trainY.shape[1]
n_hidden1 = 256
n_hidden2=128
n_hidden3=56
print(numFeatures,numLabels)

#create placeholders for data
X=tf.placeholder(tf.float32,shape=[None,numFeatures])
y_=tf.placeholder(tf.float32,shape=[None,numLabels] )
W_1=tf.Variable(tf.random_normal([numFeatures,n_hidden1]))
W_2=tf.Variable(tf.random_normal([n_hidden1,n_hidden2]))
W_3=tf.Variable(tf.random_normal([n_hidden2,n_hidden3]))
W_4=tf.Variable(tf.random_normal([n_hidden3,numLabels]))




b_1=tf.Variable(tf.random_normal([n_hidden1]))
b_2=tf.Variable(tf.random_normal([n_hidden2]))
b_3=tf.Variable(tf.random_normal([n_hidden3]))
b_4=tf.Variable(tf.random_normal([numLabels]))


sess = tf.InteractiveSession()
sess.close()
sess = tf.InteractiveSession()
#lets define the layers 
learning_rate=0.005
"""We will have:
one input layer - 30X6, two hidden layers (256 nodes, 128 nodes), sigmoid layer
"""
layer_1 = tf.matmul(X, W_1) + b_1
h_layer_1 = tf.nn.leaky_relu(layer_1,alpha=0.2)
dropout1 = tf.nn.dropout(h_layer_1,rate=0.5)
layer_2 = tf.matmul(dropout1,W_2)+b_2
h_layer_2 = tf.nn.leaky_relu(layer_2,alpha=0.2)
dropout2 = tf.nn.dropout(h_layer_2,rate=0.5)
layer_3 = tf.matmul(dropout2,W_3)+b_3
h_layer_3 = tf.nn.sigmoid(layer_3)
dropout3 = tf.nn.dropout(h_layer_3,rate=0.5)
#
output_raw = tf.matmul(dropout3, W_4)+b_4
output= tf.nn.sigmoid(output_raw)
#output_raw = tf.matmul(dropout2, W_3)+b_3
#output= tf.nn.sigmoid(output_raw)


cost = tf.reduce_mean(tf.pow(output - y_, 2))


optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

model =keras.Sequential([
        keras.layers.Dense(numFeatures),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='sigmoid'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(56, activation='sigmoid'),
        keras.layers.Dense(numLabels, activation='sigmoid'),
        ])
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'],validation_split=0.2)
###LSTM
history=model.fit(trainX,trainY,batch_size=50,verbose=1,epochs=2500)

# Generate generalization metrics
score = model.evaluate(testX, testY, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_acc'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()




type(history.history)

display_step=100
cost_val=[]
for epoch in range(25000):
    _, c = sess.run([optimizer, cost], feed_dict={X: trainX, y_: trainY})
    # Display logs per epoch step
    if epoch % display_step == 0:
        cost_val.append(c)
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))
        

print("Optimization Finished!")



predictions, train_accuracy_score=sess.run([output,accuracy],feed_dict={X:trainX, y_: trainY})
test_accuracy_score=sess.run(accuracy,feed_dict={X:testX, y_: testY})
print('train_accuracy_score',train_accuracy_score)
print('test_accuracy_score',test_accuracy_score)

plt.plot(cost_val)




#######################
#lets predict on test kaggle data
#import data
pred_data_original=pd.read_csv('test.csv')
pred_data=pred_data_original.copy()







pred_data['Embarked'].fillna(value='S',inplace=True)
pred_data['family']=pred_data['SibSp']+pred_data['Parch']+1
pred_data['Sex'] = pred_data['Sex'].replace(['female','male'],[0,1])
pred_data['Embarked'] = pred_data['Embarked'].replace(['S','Q','C'],[1,2,3])
pred_data['Title']=pred_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

pred_data['Title'] = pred_data['Title'].map(title_mapping)
#Assign median age of particular group to missing values
pred_data["Age"].fillna(pred_data.groupby("Title")["Age"].transform("median"), inplace=True)
# fill missing Fare with median fare for each Pclass
pred_data["Fare"].fillna(pred_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#cabin
pred_data['Cabin'] = pred_data['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8} #we are scaling here
pred_data['Cabin'] = pred_data['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass
pred_data["Cabin"].fillna(pred_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


pred_data = pred_data.drop(['PassengerId','Name', 'SibSp','Parch','Ticket'], axis=1)




to_pred=pred_data.to_numpy()

predictions=sess.run(output,feed_dict={X:to_pred})

df_predicted = pd.DataFrame(data=predictions, index=None, columns=["yes", "No"])
df_predicted.loc[df_predicted.yes>0.5,'yes'] = 1
df_predicted.loc[df_predicted.yes<0.5,'yes'] = 0
df_predicted.loc[df_predicted.No<0.5,'No'] = 0

df_predicted.loc[df_predicted.No>0.5,'No'] = 1
df_predicted.reset_index(inplace=True)

df_predicted.head(20)




pred_data_original.reset_index(inplace=True)


data_new= pred_data_original.merge(df_predicted,on='index')

final_predicted_data = data_new.drop(['index','Cabin','Fare','Ticket','Embarked','Name', 'Pclass', 'Sex','Age', 'SibSp','Parch','No'], axis=1)



final_predicted_data = final_predicted_data.rename(columns={'yes': 'Survived'})
final_predicted_data = final_predicted_data.astype(int) # convert all data to float before converting into a tensorflow dataset

final_predicted_data.head()

final_predicted_data.to_csv('final_predicted_data7.csv',index=False)


data_original['Fare1'] = pd.qcut(data_original['Fare'], 13)
data_original['Fare2']=pd.get_dummies(data_original)

