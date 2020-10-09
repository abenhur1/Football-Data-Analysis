import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.svm import SVC

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

SLLFilteredForML = pd.read_pickle('SLLFilteredForML.pkl')
filteredForML = pd.read_pickle('filteredForML.pkl')
filteredForML.drop(filteredForML.loc[2551:2711].index, inplace=True)
# filteredForMLHoldOut = pd.read_pickle('filteredForMLHoldOut.pkl')

## Separate into feature set and target variable, and split the dataset into training and testing set.
X = SLLFilteredForML.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
y = SLLFilteredForML['FTR'].copy()
for col in X.columns:
    X[col] = scale(X[col])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=2, stratify=y)


# Makes predictions using a fit classifier based on F1 score.
def predict_labels(classifier, features, target):
    y_pred = classifier.predict(features)

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


# Train and predict using a classifer based on F1 score.
def train_predict(classifier, X_trained, y_trained, X_tested, y_tested):
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}:".format(classifier.__class__.__name__, len(X_trained)))

    classifier.fit(X_trained, y_trained)

    # Print the results of prediction for both training and testing
    f1Score, accuracy = predict_labels(classifier, X_trained, y_trained)
    print(f1Score, accuracy)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1Score, accuracy))

    f1Score, accuracy = predict_labels(classifier, X_tested, y_tested)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1Score, accuracy))


## Initialize some models:
clf_SVC = SVC(random_state=2)
clf_xgb = xgb.XGBClassifier(seed=3)

for clf in [clf_SVC, clf_xgb]:
    train_predict(clf, X_train, y_train, X_test, y_test)
    print(clf.get_params())
    print('')
    print('')


# print('')
# print('------------MAJOR TUNING IS HAPPENING------------')
# print('')
#
# ## SVC:
# SVC_parameters = {'C': [10],
#                   'gamma': [0.01],
#                   'kernel': ['rbf']}
#
# # reinitialize the classifier
# clf_SVC = SVC(random_state=2)
# # Make an f1 scoring function using 'make_scorer'
# f1_scorer = make_scorer(f1_score, pos_label='H')
# # Perform grid search on the classifier using the f1_scorer as the scoring method
# SVC_grid_obj = GridSearchCV(clf_SVC, scoring=f1_scorer, param_grid=SVC_parameters, cv=5)
# # Fit the grid search object to the training data and find the optimal parameters
# SVC_grid_obj = SVC_grid_obj.fit(X_train, y_train)
# # Get the estimator
# clf_SVC = SVC_grid_obj.best_estimator_
# print(SVC_grid_obj.get_params())
#
# ## Reports:
# # Report the final F1 score for training and testing after parameter tuning
# f1, acc = predict_labels(clf_SVC, X_train, y_train)
# print("{} F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format('clf_SVC', f1, acc))
#
# f1, acc = predict_labels(clf_SVC, X_test, y_test)
# print("{} F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format('clf_SVC', f1, acc))
#
# print('')
#
# ## xgb:
# xgb_parameters = {'learning_rate': [0.3],
#                   'n_estimators': [40],
#                   'max_depth': [4],
#                   'min_child_weight': [5],
#                   'gamma': [1],
#                   'subsample': [0.8],
#                   'colsample_bytree': [0.8],
#                   'scale_pos_weight': [1],
#                   'reg_alpha': [1e-5]}
#
# # reinitialize the classifier
# xgb_clf = xgb.XGBClassifier(seed=2)
# # Make an f1 scoring function using 'make_scorer'
# f1_scorer = make_scorer(f1_score, pos_label='H')
# # Perform grid search on the classifier using the f1_scorer as the scoring method
# xgb_grid_obj = GridSearchCV(xgb_clf, scoring=f1_scorer, param_grid=xgb_parameters, cv=5)
# # Fit the grid search object to the training data and find the optimal parameters
# xgb_grid_obj = xgb_grid_obj.fit(X_train, y_train)
# # Get the estimator
# xgb_clf = xgb_grid_obj.best_estimator_
# print(xgb_grid_obj.get_params())
#
# ## Reports:
# # Report the final F1 score for training and testing after parameter tuning
# f1, acc = predict_labels(xgb_clf, X_train, y_train)
# print("{} F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format('xgb_clf', f1, acc))
#
# f1, acc = predict_labels(xgb_clf, X_test, y_test)
# print("{} F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format('xgb_clf', f1, acc))

# print('')
# print('------------VALIDATION HAPPENING------------')
# print('')
#
#
# ## Validation on HoldOut data:
# X_La_Liga_HO = filteredForMLHoldOut.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
# scaler = MinMaxScaler()
# X_La_Liga_HO = scaler.fit_transform(X_La_Liga_HO)
# y_La_Liga_HO = filteredForMLHoldOut['FTR'].copy()
# f1, acc = predict_labels(clf, X_La_Liga_HO, y_La_Liga_HO)
# print(X_La_Liga_HO.shape, y_La_Liga_HO.shape)
# print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
