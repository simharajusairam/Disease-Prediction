## Dealing with the Kidney Disease Dataset

# Importing of required libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Importing and reading of kidney disease dataset for predictions
df = pd.read_csv("kidney_disease.csv")

# Data Analysis and Cleaning of data
df[["htn", "dm", "cad", "pe", "ane"]] = df[["htn", "dm", "cad", "pe", "ane"]].replace(to_replace = {'yes':1, 'no':0})
df[["rbc", "pc"]] = df[["rbc", "pc"]].replace(to_replace = {"abnormal":1, "normal":0})
df[["pcc", "ba"]] = df[["pcc", "ba"]].replace(to_replace = {"present":1, "notpresent":0})
df[["appet"]] = df[["appet"]].replace(to_replace = {"good":1, "poor":0, "no":np.nan})
df["classification"] = df["classification"].replace(to_replace = {"ckd":1.0, "ckd\t":1.0, "notckd":0.0, "no":0.0})
df.rename(columns = {"classification" : "class"}, inplace = True)

# Further cleaning of data
df["pe"] = df["pe"].replace(to_replace = "good", value = 0)
df["appet"] = df["appet"].replace(to_replace = 'no', value = 0)
df["cad"] = df["cad"].replace(to_replace = "\tno", value = 0)
df["dm"] = df["dm"].replace(to_replace = {"\tno":0, "\tyes":1, " yes":1, "":np.nan})

df.drop("id", axis = 1, inplace = True)
df = df.drop(["su", "rbc", "rc", "wc", "pot", "sod"], axis = 1)
df["pcv"] = df["pcv"].fillna(method = "ffill")
df.drop(["pc"], axis = 1, inplace = True)
df["hemo"] = df["hemo"].fillna(method = "ffill")
df.drop(["sg"], axis = 1, inplace = True)

df = df.fillna(method = "ffill")

df.drop(["ba"], axis = 1, inplace = True)
df.drop(["pe"], axis = 1, inplace = True)
df.drop(["cad"], axis = 1, inplace = True)
df.drop(["ane"], axis = 1, inplace = True)

df = df.replace("\t?", 31)
print(df.columns)
print(df.shape[1])

target = df["class"]
source = df.drop(["class"], axis = 1)

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(source, target, test_size = 0.05)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

# Using of logistic regression ML algorithm for traing the model
lr = LogisticRegression()
lr.fit(X_train, y_train)

a11 = cross_validate(lr, source, target, cv = 10) 
print(a11)

# Dumping of complete file
joblib.dump(lr, "model4")

