## Dealing with the heart dataset

# Importing of required libraries
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
from sklearn.linear_model import LogisticRegression
# warnings.filterwarnings("ignore", category = DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

# Importing and reading of heart dataset for prediction
data = pd.read_csv("heart.csv")
data["trestbps"] = np.log(data["trestbps"])

# Data Analysis
data = data.drop(["fbs"], axis = 1)
data = data.drop(["ca"], axis = 1)
data["chol"] = np.log(data["chol"])
target = data["target"]
print(data.shape[1])

np.random.shuffle(data.values)
data = data.drop(["target"], axis = 1)
print(data.columns)
sc = StandardScaler()
data = sc.fit_transform(data)

# Using of logistic regression ML algorithm for training of data
lr = LogisticRegression()
lr.fit(data,target)
cv_results = cross_validate(lr, data, target, cv = 10)
print(cv_results)

# Dumping of complete file
joblib.dump(lr,"model2")

