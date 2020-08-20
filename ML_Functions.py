import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt

laLiga0919FilteredML = pd.read_pickle('laLiga0919ML.pkl')

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)


# Makes predictions using a fit classifier based on F1 score:
def predict_labels(classifier, features, target):
    y_pred = classifier.predict(features)

    return f1_score(target, y_pred, average='weighted', labels=np.unique(y_pred)), sum(target == y_pred) / float(len(y_pred))


# Train and predict using a classifer based on F1 score:
def train_predict(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(classifier.__class__.__name__, len(X_train)))

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(classifier, X_train, y_train)
    print(f1, acc)

    print("F1 score and accuracy score (before GridSearch) for training set: {:.4f} , {:.4f}.".format(f1, acc))

    f1, acc = predict_labels(classifier, X_test, y_test)
    print("F1 score and accuracy score (before GridSearch) for test set: {:.4f} , {:.4f}.".format(f1, acc))


X_La_Liga = laLiga0919FilteredML.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
print(X_La_Liga.columns)
y_La_Liga = laLiga0919FilteredML['FTR'].copy()

# scatter = scatter_matrix(X_La_Liga[['HTAggGoalScored', 'ATAggGoalScored', 'HTAggGoalConceded', 'ATAggGoalConceded',
#                                                 'HTAggLeaguePoints', 'ATAggLeaguePoints', 'NumOfPastHTSpecificWinsOutOf3',
#                                                 'NumOfPastATSpecificWinsOutOf3', 'NumOfPastHTWinsOutOfLast3Matches',
#                                                 'NumOfPastATWinsOutOfLast3Matches', 'HTWinningChancesAtHome', 'ATWinningChancesWhenAway']])
# plt.show()

X_trained, X_tested, y_trained, y_tested = train_test_split(X_La_Liga, y_La_Liga, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_trained)
X_test_scaled = scaler.fit_transform(X_tested)

clf_SVC = SVC(random_state=912, kernel='rbf')
clf_LogReg = LogisticRegression(random_state=42)
clf_XGB = xgb.XGBClassifier(seed=82)
clf_LinDiscriminantAnalysis = LinearDiscriminantAnalysis()
clf_list = [clf_SVC, clf_LogReg, clf_LinDiscriminantAnalysis, clf_XGB]

for clf in clf_list:
    train_predict(clf, X_train_scaled, y_trained, X_test_scaled, y_tested)
    print('')

# Create the parameters list you wish to tune
parameters = {'learning_rate': [0.02, 0.05, 0.1], 'n_estimators': [40, 80, 160], 'max_depth': [3, 4, 5], 'min_child_weight': [1, 3, 5, 10],
              'gamma': [0.4, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0], 'scale_pos_weight': [1],
              'reg_alpha': [1e-5]}
clf = xgb.XGBClassifier(seed=2)

# Make an f1 scoring function using 'make_scorer':
# precision_scorer = make_scorer(precision_score, average='weighted')
recall_scorer = make_scorer(recall_score, average='weighted')
f1_scorer = make_scorer(f1_score, average='weighted')

# Perform grid search on the classifier using the different scorers:
for scorer in [recall_scorer, f1_scorer]:
    grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring=scorer, cv=5)
    # Fit the grid search object to the training data and find the optimal parameters
    grid_object = grid_obj.fit(X_train_scaled, y_trained)

clf = grid_object.best_estimator_  # Get the best estimator
print(str(scorer) + ': {}'.format(clf))

# Final F1 score for training and testing after GridSearch:
f1Score, accuracy = predict_labels(clf, X_train_scaled, y_trained)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1Score, accuracy))
f1Score, accuracy = predict_labels(clf, X_test_scaled, y_tested)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1Score, accuracy))
