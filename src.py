import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv(filename)
#Split data into training and test sets.

y = data.attribute #declare the attribute to be tested against
X = data.drop(attribute, axis=1) #include chosen attribute as the first parameter

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=123, 
                                                    stratify=y)
#preprocessing data
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))
scaler = preprocessing.StandardScaler().fit(x_train)
#Applying transformer to training data
x_train_scaled = scaler.transform(x_train)
#print(X_train_scaled.std(axis=0))
#Applying transformer to test data

x_test_scaled = scaler.transform(x_test)
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

#performing crossvalidation with pipeline. keeps data wiyhin the test fold.
clf = GridSearchCV(pipeline, hyperparameters, cv=8)
 
# Fit and tune model
clf.fit(x_train, y_train)

#print(X_test_scaled)

y_pred = clf.predict(x_test)
print(r2_score(y_test, y_pred))
print mean_squared_error(y_test, y_pred) #to test the error margain

joblib.dump(clf, 'rf_regressor.pkl') #saves file

clf2 = joblib.load('rf_regressor.pkl') #loads file for future
 
# Predict data set using loaded model
#clf2.predict(x_test)
