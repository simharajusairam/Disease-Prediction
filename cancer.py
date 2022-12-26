## Dealing with the Cancer Dataset

# Importing of required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

logreg = LogisticRegression()

# importing and Reading of dataset
data = pd.read_csv("cancer.csv")
data.replace([np.inf, -np.inf], np.nan, inplace = True)
data.fillna(999, inplace = True)
# data.drop(["Unnamed : 32"], axis = 1, inplace = True)  # Dropping of unnamed column because of no use
data.drop(["id"], axis = 1, inplace = True)   # Dropping of id column because of no use

a = pd.get_dummies(data["diagnosis"])   # Converting of categorical data into numarical data by using get_dummies function from pandas
cancer = pd.concat([data, a], axis = 1)    # Concatinating of columns for calculations
cancer.drop(["diagnosis", "B"], axis = 1, inplace = True)
cancer.rename(columns = {"M":"Malignant/Benign"}, inplace = True)  # Renaming of "M" column to another name for better understanding

# Separating of independent columns (X) and dependent column (y) for prediction 
y = cancer[["Malignant/Benign"]]
X = cancer.drop(["Malignant/Benign"], axis = 1)
print(X.shape[1])

X = np.array(X)
y = np.array(y)

# Fitting the model & Using of logistic regression ML algorithm
logreg.fit(X,y.reshape(-1,))

# Dumping of complete file
joblib.dump(logreg, "model")


