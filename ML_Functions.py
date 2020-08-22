# הולד אאוט דאטה של שתי עונות אחרונות. 
# שמות של פונקציותפונקציית מיין 
# רעיונות נוספים: כמה אחוז מהנקודות הושגו בבית/אחוז נצחונות מתוך כלל המשחקים (לא רק בבית)נקודות ממושקלות לפי כמות גולים במשחק?

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt

laLiga0919FilteredMLFULLROWS = pd.read_pickle('laLiga0919MLFullRows.pkl')
laLiga0919Filtered = pd.read_pickle('laLiga0919ML.pkl')

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

# clf_SVC = SVC(random_state=912, kernel='rbf')
# clf_LogReg = LogisticRegression(random_state=42)
# clf_XGB = xgb.XGBClassifier(seed=82)
# clf_LinDiscriminantAnalysis = LinearDiscriminantAnalysis()
# clf_list = [clf_SVC, clf_LogReg, clf_LinDiscriminantAnalysis, clf_XGB]
#
# for clf in clf_list:
#     train_predict(clf, X_trained_scaled, y_trained, X_tested_scaled, y_tested)
#     print('')

X_La_Liga = laLiga0919Filtered.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
print(X_La_Liga.columns)
y_La_Liga = laLiga0919Filtered['FTR'].copy()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_La_Liga)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_La_Liga, test_size=0.2, random_state=42)

xgbc = xgb.XGBClassifier(learning_rate=0.5,
                    n_estimators=150,
                    max_depth=6,
                    min_child_weight=0,
                    gamma=0,
                    reg_lambda=1,
                    subsample=1,
                    colsample_bytree=0.75,
                    scale_pos_weight=1,
                    objective='multi:softprob',
                    num_class=3,
                    random_state=42)

mcl = xgbc.fit(X_train, y_train, eval_metric='mlogloss')
pred = mcl.predict(X_test)
proba = mcl.predict_proba(X_test)

#store our flower labels for results
y_map = pd.DataFrame(data=y_La_Liga, columns=['class'])
y_map['label'] = y_map['class'].map({0:'H',1:'D',2:'A'})
print(y_map)

print(mcl.score(X_scaled, y_La_Liga))