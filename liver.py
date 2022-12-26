## Dealing with the Liver Dataset

# Importing of required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Importing and reading of liver disease dataset for prediction
data = pd.read_csv("indian_liver_patient.csv")
data = data.fillna(method = "ffill")
data.Gender = data.Gender.map({"Female":1, "Male":0})
data["Dataset"] = data["Dataset"].map({1:0, 2:1})
np.random.shuffle(data.values)
print(data.shape[1])
print(data.columns)


target = data["Dataset"]
source = data.drop(["Dataset"], axis = 1)
sm = SMOTE()

sc = StandardScaler()
lr = LogisticRegression()  # Using of logistic regression ML algorithm for training the model
source = sc.fit_transform(source)

X_train,X_test,y_train,y_test = train_test_split(source, target, test_size = 0.01)  # Splitting of data
X_train, y_train = sm.fit_resample(X_train, y_train)

cv = cross_validate(lr, X_train, y_train, cv = 10)

# Fitting the model for prediction
lr.fit(X_train, y_train)
print(cv)

# Dumping the complete file
joblib.dump(lr, "model3")


