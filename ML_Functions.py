# import pandas as pd
# from pandas import to_numeric
# import sqlite3
# import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from Leagues_Data_and_Adaptations import X_La_Liga, y_La_Liga


def predict_labels(classifier, features, target):
    y_pred = classifier.predict(features)

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


for col in X_La_Liga.columns:
    X_La_Liga[col] = scale(X_La_Liga[col])
X_La_Liga_train, X_La_Liga_validation, y_La_Liga_train, y_La_Liga_validation = \
    train_test_split(X_La_Liga, y_La_Liga, test_size=0.20, random_state=1)

# Parameter tuning:
parameters = {'learning_rate': [0.1],
              'n_estimators': [40],
              'max_depth': [3],
              'min_child_weight': [3],
              'gamma': [0.4],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'scale_pos_weight': [1],
              'reg_alpha': [1e-5]
              }

# Evaluate each model in turn and compare algorithms:
models = [('LogReg', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LinDiscAnal', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('DeciTree', DecisionTreeClassifier()), ('GaussianNB', GaussianNB()),
          ('SVM', SVC(kernel='rbf', gamma='auto')), ('XGB', xgb.XGBClassifier())]
results = []
names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_La_Liga_train, y_La_Liga_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(xgb.XGBClassifier(), X_La_Liga_train, y_La_Liga_train, cv=kfold, scoring='accuracy')
print('%s: mean:%f (mean:%f)' % ('XGB', cv_results.mean(), cv_results.std()))
# Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='H')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_La_Liga, y_La_Liga)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_La_Liga, y_La_Liga)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_La_Liga, y_La_Liga)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
