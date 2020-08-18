import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from pandas.plotting import scatter_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score

from Data_Cleaning_for_ML import laLiga0919FilteredML

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 16)


X_La_Liga = laLiga0919FilteredML.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1)
print(X_La_Liga.head())
y_La_Liga = laLiga0919FilteredML['FTR']
# print(y_La_Liga.head())
X_train, X_test, y_train, y_test = train_test_split(X_La_Liga, y_La_Liga, random_state = 0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# for col in X_La_Liga.columns:
#     X_La_Liga[col] = scale(X_La_Liga[col])
print(X_La_Liga.head())
# X_La_Liga_train, X_La_Liga_validation, y_La_Liga_train, y_La_Liga_validation = \
#     train_test_split(X_La_Liga, y_La_Liga, test_size=0.20, random_state=1)

## Evaluate each model in turn and compare algorithms:
models = [('LogReg', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LinDiscAnal', LinearDiscriminantAnalysis()),
          ('GaussianNB', GaussianNB()), ('SVM (kernel=rbf)', SVC(kernel='rbf', gamma='auto', random_state=1)), ('XGB', xgb.XGBClassifier(seed=2))]
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_La_Liga_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
scatter_matrix(X_La_Liga[['HTAggGoalScored', 'ATAggGoalScored', 'HTAggGoalConceded', 'ATAggGoalConceded', 'HTAggLeaguePoints', 'ATAggLeaguePoints',
                          'NumOfPastHTSpecificWinsOutOf3', 'NumOfPastATSpecificWinsOutOf3', 'NumOfPastHTWinsOutOfLast3Matches',
                          'NumOfPastATWinsOutOfLast3Matches', 'HTWinningChancesAtHome', 'ATWinningChancesWhenAway']], figsize=(10, 10))
plt.figure()
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# cv_results = cross_val_score(xgb.XGBClassifier(), X_La_Liga_train, y_La_Liga_train, cv=kfold, scoring='accuracy')
# print('%s: mean:%f (mean:%f)' % ('XGB', cv_results.mean(), cv_results.std()))
