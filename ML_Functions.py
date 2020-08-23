# לחשב את נוחות בבית עד המשחק ולא כל העשר שנים

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

laLiga0919Filtered = pd.read_pickle('laLiga0919ML.pkl')
laLiga0919FilteredHoldOut = pd.read_pickle('laLiga0919MLHoldOut.pkl')

# Separate into feature set and target variable.
X_La_Liga = laLiga0919Filtered.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
print(X_La_Liga.columns)
# Center to the mean and component wise scale to unit variance.
scaler = MinMaxScaler()
X_La_Liga = scaler.fit_transform(X_La_Liga)
y_La_Liga = laLiga0919Filtered['FTR'].copy()

# Split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_La_Liga, y_La_Liga, test_size=50, random_state=2, stratify=y_La_Liga)


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


# Initialize some models:
clf_LogReg = LogisticRegression(random_state=42)
clf_SVC = SVC(random_state=912, kernel='rbf')
clf_xgb = xgb.XGBClassifier(seed=82)

train_predict(clf_LogReg, X_train, y_train, X_test, y_test)
print('')
print('')
train_predict(clf_SVC, X_train, y_train, X_test, y_test)
print('')
print('')
train_predict(clf_xgb, X_train, y_train, X_test, y_test)
print('')

print('------------MAJOR TUNING IS HAPPENING------------')
print('')

# Create the parameters list you wish to tune
parameters = {'learning_rate': [0.1],
              'n_estimators': [40],
              'max_depth': [3],
              'min_child_weight': [5],
              'gamma': [0.4],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'scale_pos_weight': [1],
              'reg_alpha': [1e-5]
              }

# Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label='H')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
