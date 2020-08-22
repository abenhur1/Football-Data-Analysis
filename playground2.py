# הולד אאוט דאטה של שתי עונות אחרונות.

# רעיונות נוספים: כמה אחוז מהנקודות הושגו בבית/אחוז נצחונות מתוך כלל המשחקים (לא רק בבית)נקודות ממושקלות לפי כמות גולים במשחק?

# # Get Goal Difference
# playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
# playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']
#
# # Diff in points
# playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
# playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']
#
# # Diff in last year positions
# playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

laLiga0919Filtered = pd.read_pickle('laLiga0919ML.pkl')

# Separate into feature set and target variable
X_La_Liga = laLiga0919Filtered.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
# Center to the mean and component wise scale to unit variance.
for col in X_La_Liga.columns:
    X_La_Liga[col] = scale(X_La_Liga[col])
y_La_Liga = laLiga0919Filtered['FTR'].copy()

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_La_Liga, y_La_Liga, test_size=50, random_state=2, stratify=y_La_Liga)


def train_classifier(classifier, X_trained, y_trained):
    """ Fits a classifier to the training data. """
    classifier.fit(X_trained, y_trained)


def predict_labels(classifier, features, target):
    """ Makes predictions using a fit classifier based on F1 score. """
    y_pred = classifier.predict(features)

    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


def train_predict(classifier, X_trained, y_trained, X_tested, y_tested):
    """ Train and predict using a classifer based on F1 score. """

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(classifier.__class__.__name__, len(X_trained)))

    # Train the classifier
    train_classifier(classifier, X_trained, y_trained)

    # Print the results of prediction for both training and testing
    f1Score, accuracy = predict_labels(classifier, X_trained, y_trained)
    print(f1Score, accuracy)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1Score, accuracy))

    f1Score, accuracy = predict_labels(classifier, X_tested, y_tested)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1Score, accuracy))


# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=912, kernel='rbf')
# Boosting refers to this general problem of producing a very accurate prediction rule
# by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed=82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print('')

# Create the parameters list you wish to tune
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
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
