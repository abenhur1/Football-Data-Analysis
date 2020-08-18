import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from time import time
from sklearn.metrics import f1_score
from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import scale
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score

from Data_Cleaning_for_ML import laLiga0919FilteredML

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 16)

## for measuring training time
# F1 score (also F-score or F-measure) is a measure of a test's accuracy.
# It considers both the precision p and the recall r of the test to compute
# the score: p is the number of correct positive results divided by the number of
# all positive results, and r is the number of correct positive results divided by
# the number of positive results that should have been returned. The F1 score can be
# interpreted as a weighted average of the precision and recall, where an F1 score
# reaches its best value at 1 and worst at 0.


def train_classifier(clf, X_train, y_train):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    """ Makes predictions using a fit classifier based on F1 score. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))

    return f1_score(target, y_pred, average='micro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    """ Train and predict using a classifer based on F1 score. """

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


X_La_Liga = laLiga0919FilteredML.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
print('X_La_Liga head:', X_La_Liga.head())
y_La_Liga = laLiga0919FilteredML['FTR'].copy()
print('y_La_Liga head:', y_La_Liga.head())

scatter_matrix(X_La_Liga[['HTAggGoalScored', 'ATAggGoalScored', 'HTAggGoalConceded', 'ATAggGoalConceded', 'HTAggLeaguePoints', 'ATAggLeaguePoints',
                          'NumOfPastHTSpecificWinsOutOf3', 'NumOfPastATSpecificWinsOutOf3', 'NumOfPastHTWinsOutOfLast3Matches',
                          'NumOfPastATWinsOutOfLast3Matches', 'HTWinningChancesAtHome', 'ATWinningChancesWhenAway']], figsize=(10, 10))

X_trained, X_tested, y_trained, y_tested = train_test_split(X_La_Liga, y_La_Liga, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_trained)
X_test_scaled = scaler.transform(X_tested)
# for col in X_La_Liga.columns:
#     X_La_Liga[col] = scale(X_La_Liga[col])
print(X_La_Liga.head())
print(X_trained)


# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=912, kernel='rbf')
# Boosting refers to this general problem of producing a very accurate prediction rule
# by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed=82)

train_predict(clf_A, scaler.fit_transform(X_trained), y_trained, scaler.transform(X_tested), y_tested)
print('')
train_predict(clf_B, scaler.fit_transform(X_trained), y_trained, scaler.transform(X_tested), y_tested)
print('')
train_predict(clf_C, scaler.fit_transform(X_trained), y_trained, scaler.transform(X_tested), y_tested)
print('')
