
#LEARNING SVM MODEL

import numpy as np #To create numpy arrays
import pandas as pd #To create dataframes
from sklearn.preprocessing import StandardScaler #To Standardize Data
from sklearn.model_selection import train_test_split #To split data into train and test
from sklearn import svm #To get the SVM Algorithm
from sklearn.metrics import accuracy_score #To check the accuracy of the model

#First create a dataframe of your data

diabetes_dataset = pd.read_csv("C:/Programming/Python/Machine Learning/resources/diabetes.csv")
#print(diabetes_dataset.head())

# get statistical measured of the data
diabetes_dataset.describe()

'''
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
### x% of values are less than the given number in the cell ###
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000
'''

diabetes_dataset['Outcome'].value_counts()
#used to get count of different values in a dataframe
'''
Outcome
0    500
1    268
'''

diabetes_dataset.groupby('Outcome').mean()
#groups mean values of all columns w.r.t. different elements in outcome

'''
         Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin        BMI  DiabetesPedigreeFunction        Age
Outcome
0           3.298000  109.980000      68.184000      19.664000   68.792000  30.304200                  0.429734  31.190000
1           4.865672  141.257463      70.824627      22.164179  100.335821  35.142537                  0.550500  37.067164
'''

X = diabetes_dataset.drop(columns = "Outcome", axis = 1)
#Drops the column Outcomes so contains everything except the outcome, X is the features column
#The input variables used to predict
Y = diabetes_dataset["Outcome"]
#Puts outcomes as the only column, Y is called the target column
#The output variable needed


#DATA STANDARDIZATION#


### converting all the data to a standard range which helps the machine learning model to make better predictions ###

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
### transforming all the data to a common range using StandardScaler ###
### can also use scaler.fit_transform to do both fit and transform in a single step ###

standardized_data
'''
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]
 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...
 
 all of it is in the range 0 and 1
'''

X = standardized_data

### Splitting the data into train and test sets ###

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state= 2)
# X = Input data used to make predictions
# Y = Output data to achieve after prediction
# test_size = fraction of the dataset to be used as training data (here 20% of dataset will be used for testing)
# stratify = used to give equal proportions of outcomes to training and testing like doesnt assign more 0 outcomes to train and more 1 outcomes to test which ensures prediction is unbiased
# random_state = ensures reproducibility, assigns a seed to the RNG which basically means randomly selecting the training and testing data by seed


# TRAINING THE MODEL #

### training SVM model ###
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
### fitting training data into classifier to train model ###

prediction = classifier.predict(X_train)
trainingAccuracy = accuracy_score(prediction, Y_train)
### comparing the predictions of training and comparing it to Outcomes to get the accuracy of the model ###

#print("Accuracy: ", trainingAccuracy*100, "%")
'''Accuracy:  78.66449511400651 %'''
### Accuracy > 75% is considered good and optimizers can be used to increase it later on ###

### checking the unknown data that the model has not trained from ###
testPrediction = classifier.predict(X_test)
testAccuracy = accuracy_score(testPrediction, Y_test)

#print("Accuracy: ", testAccuracy*100, "%")
''' Accuracy:  77.27272727272727 % '''

# CREATING A PREDICTIVE SYSTEM #

input_data = (0,162,76,56,100,53.2,0.759,25)

dataArray = np.asarray(input_data)
reshapedData = dataArray.reshape(1,-1)
std_data = scaler.transform(reshapedData)
print(std_data, "\n")

prediction = classifier.predict(std_data)
print(prediction)