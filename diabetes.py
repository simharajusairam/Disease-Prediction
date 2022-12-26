## Dealing with the Diabetes Dataset

# Importing of Required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Importing and Reading of diabetes dataset for prediction
data = pd.read_csv("diabetes.csv")
print(data.head())

# Using of Logistic regression ML algorithm
logreg = LogisticRegression()

# Separating of independent columns (X) and dependent column (y)
X = data.iloc[:,:8]
print(X.shape[1])
y = data[["Outcome"]]

X = np.array(X)
y = np.array(y)

# Fitting the model
logreg.fit(X,y.reshape(-1,))

# Dumping the complete file
joblib.dump(logreg, "model1")

