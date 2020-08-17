# import pandas as pd
# from pandas import to_numeric
# import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
# from pandas.plotting import scatter_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score

from Data_Cleaning_for_ML import X_La_Liga, y_La_Liga


for col in X_La_Liga.columns:
    X_La_Liga[col] = scale(X_La_Liga[col])
print(X_La_Liga.head())
X_La_Liga_train, X_La_Liga_validation, y_La_Liga_train, y_La_Liga_validation = \
    train_test_split(X_La_Liga, y_La_Liga, test_size=0.20, random_state=1)

## Evaluate each model in turn and compare algorithms:
models = [('LogReg', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LinDiscAnal', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('DeciTree', DecisionTreeClassifier()), ('GaussianNB', GaussianNB()),
          ('SVM (kernel=rbf', SVC(kernel='rbf', gamma='auto')), ('XGB', xgb.XGBClassifier())]
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_La_Liga_train, y_La_Liga_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


# kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# cv_results = cross_val_score(xgb.XGBClassifier(), X_La_Liga_train, y_La_Liga_train, cv=kfold, scoring='accuracy')
# print('%s: mean:%f (mean:%f)' % ('XGB', cv_results.mean(), cv_results.std()))
