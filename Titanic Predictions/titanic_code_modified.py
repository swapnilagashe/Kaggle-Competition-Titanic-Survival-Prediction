#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:02:50 2020

@author: swapnillagashe
"""

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


import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
sc=StandardScaler()
input_data[['Age','Fare']] = sc.fit_transform(input_data[['Age', 'Fare']])

input_data=input_data.to_numpy()
labels_data=labels_data.to_numpy()







trainX, testX, trainY, testY = train_test_split(input_data, labels_data, test_size=0.001, random_state=42)

#parameters
numFeatures=trainX.shape[1]
numLabels = trainY.shape[1]
n_hidden1 = 256
n_hidden2=128
n_hidden3=56
print(numFeatures,numLabels)
num_folds = 10
# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((trainX, testX), axis=0)
targets = np.concatenate((trainY, testY), axis=0)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
batch_size=50
no_epochs=1000
verbosity=1
validation_split=0.2

for train, test in kfold.split(trainX, trainY):
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
    
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    history=model.fit(inputs[train],targets[train],batch_size=batch_size, epochs=no_epochs,verbose=verbosity,                                         validation_split=validation_split)
    
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


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
    
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

model.fit(trainX,trainY,batch_size=batch_size, epochs=no_epochs,verbose=verbosity,                                         validation_split=validation_split)

model.evaluate(testX, testY, verbose=0)

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

sc=StandardScaler()
pred_data[['Age','Fare']] = sc.fit_transform(pred_data[['Age', 'Fare']])


to_pred=pred_data.to_numpy()



predictY=model.predict(to_pred)
predicted= pd.DataFrame(np.argmax(predictY,1))
predicted=1-predicted
pred_data_original.reset_index(inplace=True)
predicted.reset_index(inplace=True)

data_new= pred_data_original.merge(predicted,on='index')
final_predicted_data = data_new.drop(['index','Cabin','Fare','Ticket','Embarked','Name', 'Pclass', 'Sex','Age', 'SibSp','Parch'], axis=1)



final_predicted_data = final_predicted_data.rename(columns={0: 'Survived'})
final_predicted_data = final_predicted_data.astype(int) # convert all data to float before converting into a tensorflow dataset

final_predicted_data.head()

final_predicted_data.to_csv('final_final_predicted_data_withKfold2.csv',index=False)
