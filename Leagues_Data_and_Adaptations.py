import pandas as pd
import sqlite3
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import to_numeric


# import numpy as np

# notes on files: http://www.football-data.co.uk/notes.txt:
# Div = League Division
# Date = Match Date (dd/mm/yy)
# Time = Time of match kick off
# HomeTeam = Home Team
# AwayTeam = Away Team
# FTHG and HG = Full Time Home Team Goals
# FTAG and AG = Full Time Away Team Goals
# FTR and Res = Full Time Result (H=Home Win, D=Draw, A=Away Win)
# HTHG = Half Time Home Team Goals
# HTAG = Half Time Away Team Goals
# HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
#
# Match Statistics (where available)
# Attendance = Crowd Attendance
# Referee = Match Referee
# HS = Home Team Shots
# AS = Away Team Shots
# HST = Home Team Shots on Target
# AST = Away Team Shots on Target
# HHW = Home Team Hit Woodwork
# AHW = Away Team Hit Woodwork
# HC = Home Team Corners
# AC = Away Team Corners
# HF = Home Team Fouls Committed
# AF = Away Team Fouls Committed
# HFKC = Home Team Free Kicks Conceded
# AFKC = Away Team Free Kicks Conceded
# HO = Home Team Offsides
# AO = Away Team Offsides
# HY = Home Team Yellow Cards
# AY = Away Team Yellow Cards
# HR = Home Team Red Cards
# AR = Away Team Red Cards
# HBP = Home Team Bookings Points (10 = yellow, 25 = red)
# ABP = Away Team Bookings Points (10 = yellow, 25 = red)

#### Analysis Stage:
### Function Helpers:
def df_creator(path, file):
    file = pd.read_csv(path + file)
    return file


### Reading the La Liga data files and concatenate the DFs:
la_liga_path = "C:/Users/User/PycharmProjects/Football-Data-Analysis/"
files_list = ['season-0910_csv.csv',
              'season-1011_csv.csv',
              'season-1112_csv.csv',
              'season-1213_csv.csv',
              'season-1314_csv.csv',
              'season-1415_csv.csv',
              'season-1516_csv.csv',
              'season-1617_csv.csv',
              'season-1718_csv.csv',
              'season-1819_csv.csv']
laLiga0919Concat = pd.concat([df_creator(la_liga_path, file) for file in files_list])

relevant_analysis_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
## Modifying the DF:
# Leave relevant columns:
laLiga0919Filtered = laLiga0919Concat[relevant_analysis_cols].copy()
# la_liga_0919_df['Year'] = pd.DatetimeIndex(la_liga_0919_df['Date']).year  # year column.

# Filter out games that draw at HT:
laLiga0919Filtered2 = laLiga0919Filtered[((laLiga0919Filtered.HTR == 'H')
                                          | (laLiga0919Filtered.HTR == 'A'))].copy()  # Filter out games that draw at HT

# Filter out games that draw at HT and leader leads by exactly 1:
laLiga0919Filtered3 = laLiga0919Filtered[((laLiga0919Filtered.HTR == 'H')
                                          | (laLiga0919Filtered.HTR == 'A'))].copy()
laLiga0919Filtered3 = laLiga0919Filtered3[abs(to_numeric(laLiga0919Filtered3.HTHG) - to_numeric(laLiga0919Filtered3.HTAG))
                                          == 1].copy()  # Leader leads by exactly 1

# Filter out games that draw at HT and leader leads by more than 1:
laLiga0919Filtered4 = laLiga0919Filtered[((laLiga0919Filtered.HTR == 'H')
                                          | (laLiga0919Filtered.HTR == 'A'))].copy()
laLiga0919Filtered4 = laLiga0919Filtered4[abs(to_numeric(laLiga0919Filtered4.HTHG) - to_numeric(laLiga0919Filtered4.HTAG))
                                          > 1].copy()  # Leader leads by more than 1

### Master Premier League df extracted:
con = sqlite3.connect("C:/Users/User/PycharmProjects/Football-Data-Analysis/EPL_Seasons_1993-2017_RAW_Table.sqlite")
dfRawTable = pd.read_sql_query("SELECT * FROM EPL", con)

## Modifying the DF:
# Leave relevant columns:
premierLeague9518Filtered = dfRawTable[924:][relevant_analysis_cols].copy()
# premierLeague9518Filtered['Year'] = pd.DatetimeIndex(la_liga_0919_df['Date']).year  # year column.

# Filter out games that draw at HT:
premierLeague9518Filtered2 = premierLeague9518Filtered[((premierLeague9518Filtered.HTR == 'H') |
                                                        (premierLeague9518Filtered.HTR == 'A'))].copy()  # Filter out games that draw at HT
premierLeague9518Filtered2.reset_index(drop=True, inplace=True)

# Filter out games that draw at HT and leader leads by exactly 1:
premierLeague9518Filtered3 = premierLeague9518Filtered[((premierLeague9518Filtered.HTR == 'H') |
                                                        (premierLeague9518Filtered.HTR == 'A'))].copy()
premierLeague9518Filtered3 = premierLeague9518Filtered3[abs(to_numeric(premierLeague9518Filtered3.HTHG) -
                                                            to_numeric(premierLeague9518Filtered3.HTAG))
                                                        == 1].copy()  # Leader leads by exactly 1
premierLeague9518Filtered3.reset_index(drop=True, inplace=True)

# Filter out games that draw at HT and leader leads by more than 1:
premierLeague9518Filtered4 = premierLeague9518Filtered[((premierLeague9518Filtered.HTR == 'H') |
                                                        (premierLeague9518Filtered.HTR == 'A'))].copy()
premierLeague9518Filtered4 = premierLeague9518Filtered4[abs(to_numeric(premierLeague9518Filtered4.HTHG) -
                                                            to_numeric(premierLeague9518Filtered4.HTAG))
                                                        > 1].copy()
premierLeague9518Filtered4.reset_index(drop=True, inplace=True)


#### ML Stage:
### La Liga:
laLiga0919FilteredML = laLiga0919Concat.copy()
laLiga0919FilteredML.reset_index(drop=True, inplace=True)
laLiga0919FilteredML.drop(laLiga0919FilteredML.loc[:, 'B365H':'PSCA'].columns, axis=1, inplace=True)
laLiga0919FilteredML.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'HTR'], axis=1, inplace=True)
print(laLiga0919FilteredML.columns)

X_La_Liga = laLiga0919FilteredML.drop(['FTR'], axis=1)
y_La_Liga = laLiga0919FilteredML['FTR']
print(X_La_Liga.head())
for col in X_La_Liga.columns:
    X_La_Liga[col] = scale(X_La_Liga[col])
print(X_La_Liga.head())

# ### Premier League:
# premierLeague9518FilteredML = dfRawTable[924:].copy()
# premierLeague9518FilteredML.reset_index(drop=True, inplace=True)
# premierLeague9518FilteredML.drop(premierLeague9518FilteredML.loc[:, 'B365H':'B365AH'].columns, axis=1, inplace=True)
# print(premierLeague9518FilteredML.head())
#
# print(dfRawTable.columns)