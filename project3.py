# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:07:41 2020

@author: DELL
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df=pd.read_csv(r"C:\Users\DELL\Documents\cbb.csv")

df['windex'] = np.where(df.WAB > 7, 'True', 'False')

df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()

df1['POSTSEASON'].value_counts()

import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)

df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()

X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]

y = df1['POSTSEASON'].values
y[0:5]

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
tp=LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
tp

ycap=tp.predict(X_val)
ycap

ycap_prob=tp.predict(X_val)
ycap_prob

print("Train set accuracy: ", metrics.accuracy_score(y_train, tp.predict(X_train)) )
print("Validation Set accuracy: ", metrics.accuracy_score(y_val,ycap_prob) )

from sklearn.metrics import f1_score
# for f1_score please set the average parameter to 'micro'
from sklearn.metrics import log_loss

def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1
    
test_df=pd.read_csv(r"C:\Users\DELL\Documents\cbb.csv")
test_df.head()

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]

test_Feature = test_df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
test_Feature['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]

test_y = test_df1['POSTSEASON'].values
test_y[0:5]

print("Test Jaccard set accuracy: ", jaccard_index(tp.predict(test_X),test_y) )
print("Test log loss accuracy", log_loss(test_y, tp.predict_proba(test_X)))
print("f1_score accuracy:", f1_score(test_y, tp.predict(test_X), average='micro'))
