import pandas as pd
import time

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 250)


### Functions:
def create_df(path, file):
    file = pd.read_csv(path + file)
    return file


def main_func(list_of_seasons, drop_first=True):
    start = time.time()

    for season in list_of_seasons:
        update_season_df_with_teams_points_col(season)
        update_season_df_with_agg_goals_cols(season)

    league_concat = pd.concat(file for file in list_of_seasons)
    reset_index_df(league_concat)

    update_concat_df_with_last_3_specific_FTRs_cols(league_concat)
    update_concat_df_with_last_3_any_FTRs_cols(league_concat)
    update_concat_df_with_team_location_influence(league_concat)
    league_concat = drop_unnecessary_cols(league_concat)
    league_concat['FTR'] = league_concat['FTR'].apply(leave_only_Home_VS_NotHome)

    if drop_first:
        league_concat = drop_none_informative_rows(league_concat, drop_first=True, drop_none_informative=True)
    else:
        league_concat = drop_none_informative_rows(league_concat, drop_first=False, drop_none_informative=True)

    reset_index_df(league_concat)

    end = time.time()
    print(end - start)

    return league_concat


def reset_index_df(dataframe):
    return dataframe.reset_index(drop=True, inplace=True)


def rename_leagues_columns(league_df, dictionary):
    league_df.rename(columns=dictionary, inplace=True)


# Drop games from beginning of leagues and especially the first league - they never supply enough data.
def drop_none_informative_rows(league_df, drop_first=False, drop_first_num=190, drop_none_informative=False):
    if drop_first:
        league_df = league_df.iloc[drop_first_num:]

    reset_index_df(league_df)

    if drop_none_informative:
        league_df = league_df[(league_df['HTAggLeaguePointsMean'] != 0) & (league_df['ATAggLeaguePointsMean'] != 0)]

    return league_df


# This function turns ML problem from multi-class into binary
def leave_only_Home_VS_NotHome(string):
    if string == 'H':
        return 'H'
    else:
        return 'NotH'


# Some functions create columns on specific teams, they are redundant. Function is needed at the end:
def drop_unnecessary_cols(seasons_matches):
    # Drop unnecessary columns:
    for col in seasons_matches.columns:
        for team in seasons_matches.groupby('HomeTeam').median().T.columns:
            if team in col:  # the moment we encounter a column that is specific for a team, we get rid of it
                seasons_matches = seasons_matches.drop([col, ], axis=1)
                break
    return seasons_matches


# Creates a column of teams' mean scored/conceded goals UNTIL current match:
def update_season_df_with_agg_goals_cols(season_matches):
    # New columns initialized.
    season_matches['HTAggGoalScoredMean'] = 0
    season_matches['HTAggGoalConcededMean'] = 0
    season_matches['ATAggGoalScoredMean'] = 0
    season_matches['ATAggGoalConcededMean'] = 0

    for team in season_matches.groupby('HomeTeam').median().T.columns:  # A way to iterate over teams
        # Iterated by team, now we mask original df for aggregating goals. A new column is created for these values for each team. After loop,
        # we set these value to original df by making a new column as sum of all these columns. On the way we actually compute and insert their means.
        season_matches_of_team_when_home = season_matches[season_matches['HomeTeam'] == team].copy()
        season_matches_of_team_when_home['game_serial_num'] = range(0, 19)
        season_matches[team + 'asHTScoredMean'] = (season_matches_of_team_when_home['FTHG'].cumsum() -
                                                   season_matches_of_team_when_home['FTHG']) / \
                                                   season_matches_of_team_when_home['game_serial_num']  # minus FTHG values because of data leakage
        season_matches[team + 'asHTConcededMean'] = (season_matches_of_team_when_home['FTAG'].cumsum() -
                                                     season_matches_of_team_when_home['FTAG']) / \
                                                     season_matches_of_team_when_home['game_serial_num']

        season_matches_of_team_when_away = season_matches[season_matches['AwayTeam'] == team].copy()
        season_matches_of_team_when_away['game_serial_num'] = range(0, 19)
        season_matches[team + 'asATScoredMean'] = (season_matches_of_team_when_away['FTAG'].cumsum() -
                                                   season_matches_of_team_when_away['FTAG']) / \
                                                   season_matches_of_team_when_away['game_serial_num']
        season_matches[team + 'asATConcededMean'] = (season_matches_of_team_when_away['FTHG'].cumsum() -
                                                     season_matches_of_team_when_away['FTHG']) / \
                                                     season_matches_of_team_when_away['game_serial_num']

    season_matches.fillna(0, inplace=True)

    for col in season_matches.columns:
        if 'asHTScoredMean' in col:
            season_matches['HTAggGoalScoredMean'] += season_matches[col]
        elif 'asHTConcededMean' in col:
            season_matches['HTAggGoalConcededMean'] += season_matches[col]
        elif 'asATScoredMean' in col:
            season_matches['ATAggGoalScoredMean'] += season_matches[col]
        elif 'asATConcededMean' in col:
            season_matches['ATAggGoalConcededMean'] += season_matches[col]

    return season_matches


# Creates a column of teams' points UNTIL current match:
def update_season_df_with_teams_points_col(season_matches, winner_points=3):
    # New columns initialized and remapping dictionaries initialized.
    season_matches['HTAggLeaguePointsMean'] = 0
    HomeTeam_dict = {'H': winner_points, 'D': 1, 'A': 0}

    season_matches['ATAggLeaguePointsMean'] = 0
    AwayTeam_dict = {'A': winner_points, 'D': 1, 'H': 0}

    for team in season_matches.groupby('HomeTeam').median().T.columns:  # A way to iterate over teams
        # Iterated by team, now we mask original df for aggregating league points. A new column is created for these values for each team. After loop,
        # we set these value to original df by making a new column as sum of all these columns. On the way we actually compute and insert their means.
        season_matches_of_team_when_home = season_matches[season_matches['HomeTeam'] == team].copy()
        season_matches_of_team_when_home[team + 'AsHTPointReceived'] = season_matches_of_team_when_home['FTR']  # Duplicating column to remap it.
        season_matches_of_team_when_home[team + 'AsHTPointReceived'] = season_matches_of_team_when_home[team + 'AsHTPointReceived'].map(HomeTeam_dict)
        season_matches_of_team_when_home['game_serial_num'] = range(0, 19)
        season_matches[team + 'AsHTAggPointReceivedMean'] = (season_matches_of_team_when_home[team + 'AsHTPointReceived'].cumsum() -
                                                             season_matches_of_team_when_home[team + 'AsHTPointReceived']) / \
                                                             season_matches_of_team_when_home['game_serial_num']  # minus FTHG values because data leakage

        season_matches_of_team_when_away = season_matches[season_matches['AwayTeam'] == team].copy()
        season_matches_of_team_when_away[team + 'AsATPointReceived'] = season_matches_of_team_when_away['FTR']
        season_matches_of_team_when_away[team + 'AsATPointReceived'] = season_matches_of_team_when_away[team + 'AsATPointReceived'].map(AwayTeam_dict)
        season_matches_of_team_when_away['game_serial_num'] = range(0, 19)
        season_matches[team + 'AsATAggPointReceivedMean'] = (season_matches_of_team_when_away[team + 'AsATPointReceived'].cumsum() -
                                                             season_matches_of_team_when_away[team + 'AsATPointReceived']) / \
                                                             season_matches_of_team_when_away['game_serial_num']

    season_matches.fillna(0, inplace=True)

    for col in season_matches.columns:
        if 'AsHTAggPointReceivedMean' in col:
            season_matches['HTAggLeaguePointsMean'] += season_matches[col]
        elif 'AsATAggPointReceivedMean' in col:
            season_matches['ATAggLeaguePointsMean'] += season_matches[col]

    return season_matches


# Creates a column of teams' number of wins on last 3 matches between the two:
def update_concat_df_with_last_3_specific_FTRs_cols(seasons_matches):  # Notice that applies for CONCATENATED df
    num_of_matches = len(seasons_matches)
    seasons_matches['NumOfPastHTSpecificWinsOutOf3'] = 0
    seasons_matches['NumOfPastATSpecificWinsOutOf3'] = 0

    # Compute the relevant values and set them in the season_matches df:
    for general_match_ind in range(num_of_matches):
        HT = seasons_matches.iloc[general_match_ind]['HomeTeam']  # Home Team of current match
        AT = seasons_matches.iloc[general_match_ind]['AwayTeam']
        HT_win_count = 0
        AT_win_count = 0
        history_monitor = 0  # Monitors whether we reached the three past games we want to take into account

        for match_ind_until_general in range(general_match_ind - 1, -1, -1):  # To iterate backwards and find last three relevant games. It
            # happens so that it skips first game but it doesn't matter since
            # both values are of course 0
            HT_past_match = seasons_matches.iloc[match_ind_until_general]['HomeTeam']  # Home Team of past match
            AT_past_match = seasons_matches.iloc[match_ind_until_general]['AwayTeam']
            FTR_past_match = seasons_matches.iloc[match_ind_until_general]['FTR']
            if (HT in [HT_past_match, AT_past_match]) and (AT in [HT_past_match, AT_past_match]):  # To stop at relevant game
                # Above condition in order to find relevant past game
                if FTR_past_match == 'H':
                    HT_win_count = HT_win_count + 1
                elif FTR_past_match == 'A':
                    AT_win_count = AT_win_count + 1
                history_monitor = history_monitor + 1
            if history_monitor == 3:
                break  # Stop when 3 games were taken into account or before that when we reached beginning of data

        seasons_matches.at[general_match_ind, 'NumOfPastHTSpecificWinsOutOf3'] = HT_win_count  # resets value in df
        seasons_matches.at[general_match_ind, 'NumOfPastATSpecificWinsOutOf3'] = AT_win_count

    return seasons_matches


# Creates a column of teams' number of wins on their last 3 matches:
def update_concat_df_with_last_3_any_FTRs_cols(seasons_matches):  # Notice that applies for CONCATENATED df
    num_of_matches = len(seasons_matches)
    seasons_matches['NumOfPastHTWinsOutOfLast3Matches'] = 0
    seasons_matches['NumOfPastATWinsOutOfLast3Matches'] = 0

    # Compute the relevant values and set them in the season_matches df:
    for general_match_ind in range(num_of_matches):
        HT = seasons_matches.iloc[general_match_ind]['HomeTeam']  # Home Team of current match
        AT = seasons_matches.iloc[general_match_ind]['AwayTeam']
        HT_win_count = 0
        AT_win_count = 0
        HT_history_monitor = 0  # Monitors whether we reached the three past games we want to take into account
        AT_history_monitor = 0

        for match_ind_until_general in range(general_match_ind - 1, -1, -1):  # To iterate backwards and find last three relevant games. It
            # happens so that it skips first game but it doesn't matter since
            # both values are of course 0
            HT_past_match = seasons_matches.iloc[match_ind_until_general]['HomeTeam']  # Home Team of past match
            AT_past_match = seasons_matches.iloc[match_ind_until_general]['AwayTeam']
            FTR_past_match = seasons_matches.iloc[match_ind_until_general]['FTR']
            if HT in [HT_past_match, AT_past_match]:  # To stop at relevant game
                if ((HT == HT_past_match and FTR_past_match == 'H') or (HT == AT_past_match and FTR_past_match == 'A')) and HT_history_monitor < 3:
                    HT_win_count = HT_win_count + 1
                HT_history_monitor = HT_history_monitor + 1
            if AT in [HT_past_match, AT_past_match]:  # Not elif because theoretically last game could be between both
                if ((AT == HT_past_match and FTR_past_match == 'H') or (AT == AT_past_match and FTR_past_match == 'A')) and AT_history_monitor < 3:
                    AT_win_count = AT_win_count + 1
                AT_history_monitor = AT_history_monitor + 1

            if HT_history_monitor >= 3 and AT_history_monitor >= 3:
                break  # Stop when 3 games were taken into account or before that when we reached beginning of data

        seasons_matches.at[general_match_ind, 'NumOfPastHTWinsOutOfLast3Matches'] = HT_win_count  # resets value in df
        seasons_matches.at[general_match_ind, 'NumOfPastATWinsOutOfLast3Matches'] = AT_win_count

    return seasons_matches


# Creates columns of match's whereabouts' influence on FTR. If team x is at home then HTWinningChancesAtHome column's value is its num of games won
# at home divided by num of its games played at home. Similarly for that HT team, we have columns HTLosingChancesAtHome and HTDrawChancesAtHome
# function update_concat_df_with_percent_of_wins_by_location will be redundant.
def update_concat_df_with_team_location_influence(seasons_matches):  # Notice that applies for CONCATENATED df
    seasons_matches['HTWinningChancesAtHome'] = 0
    seasons_matches['ATWinningChancesWhenAway'] = 0

    groupByHTdf = seasons_matches.groupby('HomeTeam')
    groupByATdf = seasons_matches.groupby('AwayTeam')
    # numOfTeamGamesPlayed = len(groupByHTdf.get_group((list(groupByHTdf.groups)[0])))

    for key, item in groupByHTdf:
        HTCol = seasons_matches['HomeTeam']
        numOfHTTeamGamesPlayedOverall = len(item)
        numOfHTGamesWonAtHome = item[item['FTR'] == 'H'].shape[0]
        seasons_matches.loc[HTCol == key, 'HTWinningChancesAtHome'] = numOfHTGamesWonAtHome / numOfHTTeamGamesPlayedOverall

    for key, item in groupByATdf:
        ATCol = seasons_matches['AwayTeam']
        numOfATTeamGamesPlayedOverall = len(item)
        numOfATGamesWonWhenAway = item[item['FTR'] == 'A'].shape[0]
        seasons_matches.loc[ATCol == key, 'ATWinningChancesWhenAway'] = numOfATGamesWonWhenAway / numOfATTeamGamesPlayedOverall

    return seasons_matches


### Reading the La Liga data files and concatenate the DFs:
SLL_path = "C:/Users/User/PycharmProjects/Football-Data-Analysis/La Liga/"
EPL_path = "C:/Users/User/PycharmProjects/Football-Data-Analysis/Premier League/"
ML_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

### La Liga df modification:
SLL_0910_filtered_ML = create_df(SLL_path, 'season-0910_csv.csv')[ML_cols].copy()  # Every file separately because some functions are per league
SLL_1011_filtered_ML = create_df(SLL_path, 'season-1011_csv.csv')[ML_cols].copy()
SLL_1112_filtered_ML = create_df(SLL_path, 'season-1112_csv.csv')[ML_cols].copy()
SLL_1213_filtered_ML = create_df(SLL_path, 'season-1213_csv.csv')[ML_cols].copy()
SLL_1314_filtered_ML = create_df(SLL_path, 'season-1314_csv.csv')[ML_cols].copy()
SLL_1415_filtered_ML = create_df(SLL_path, 'season-1415_csv.csv')[ML_cols].copy()
SLL_1516_filtered_ML = create_df(SLL_path, 'season-1516_csv.csv')[ML_cols].copy()
SLL_1617_filtered_ML = create_df(SLL_path, 'season-1617_csv.csv')[ML_cols].copy()
SLL_1718_filtered_ML = create_df(SLL_path, 'season-1718_csv.csv')[ML_cols].copy()
SLL_1819_filtered_ML = create_df(SLL_path, 'season-1819_csv.csv')[ML_cols].copy()
SLL_1920_filtered_ML = create_df(SLL_path, 'season-1920_csv.csv')[ML_cols].copy()
EPL_0910_filtered_ML = create_df(EPL_path, 'season-0910.csv')[ML_cols].copy()
EPL_1011_filtered_ML = create_df(EPL_path, 'season-1011.csv')[ML_cols].copy()
EPL_1112_filtered_ML = create_df(EPL_path, 'season-1112.csv')[ML_cols].copy()
EPL_1213_filtered_ML = create_df(EPL_path, 'season-1213.csv')[ML_cols].copy()
EPL_1314_filtered_ML = create_df(EPL_path, 'season-1314.csv')[ML_cols].copy()
# EPL_1415_filtered_ML = create_df(EPL_path, 'season-1415.csv')[ML_cols].copy()  # PROBLEMATIC FILE APPARENTLY
EPL_1516_filtered_ML = create_df(EPL_path, 'season-1516.csv')[ML_cols].copy()
EPL_1617_filtered_ML = create_df(EPL_path, 'season-1617.csv')[ML_cols].copy()
EPL_1718_filtered_ML = create_df(EPL_path, 'season-1718.csv')[ML_cols].copy()
EPL_1819_filtered_ML = create_df(EPL_path, 'season-1819.csv')[ML_cols].copy()
EPL_1920_filtered_ML = create_df(EPL_path, 'season-1920.csv')[ML_cols].copy()


## Files' 3 lists. Experiment, Train and Test, Hold out.
experiment_list = [EPL_0910_filtered_ML, EPL_1011_filtered_ML, SLL_0910_filtered_ML, SLL_1011_filtered_ML]
seasonsFilteredList = [SLL_0910_filtered_ML,
                       SLL_1011_filtered_ML,
                       SLL_1112_filtered_ML,
                       SLL_1213_filtered_ML,
                       SLL_1314_filtered_ML,
                       SLL_1516_filtered_ML,
                       SLL_1617_filtered_ML,
                       SLL_1718_filtered_ML,
                       EPL_0910_filtered_ML,
                       EPL_1011_filtered_ML,
                       EPL_1112_filtered_ML,
                       EPL_1213_filtered_ML,
                       EPL_1314_filtered_ML,
                       EPL_1516_filtered_ML,
                       EPL_1617_filtered_ML,
                       EPL_1718_filtered_ML]
filteredListHoldOutData = [SLL_1819_filtered_ML, SLL_1920_filtered_ML, EPL_1819_filtered_ML, EPL_1920_filtered_ML]


# ## EXPERIMENT DATA:
# filteredForMLExperiment = main_func(experiment_list, drop_first=False)
## TRAIN AND TEST LIST:
filteredForML = main_func(seasonsFilteredList)
## HOLDOUT DATA:
filteredForMLHoldOut = main_func(filteredListHoldOutData, drop_first=False)

filteredForML.to_pickle('filteredForML.pkl')
filteredForMLHoldOut.to_pickle('filteredForMLHoldOut.pkl')




# להתחשב בסיכויי הימורים של החברות!

# להעמיס את קבצי הליגה האנגלית (קבצי אקסל מאותו אתר) ולהגדיל את קבוצת הטסט (כרגע היא 50 בלבד) וגם להוריד יותר שורות לא אינפורמטיביות (כרגע רק 10 יורדות)


import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

laLiga0919Filtered = pd.read_pickle('laLiga0919ML.pkl')
laLiga0919FilteredHoldOut = pd.read_pickle('laLiga0919MLHoldOut.pkl')

## Separate into feature set and target variable, and split the dataset into training and testing set.
X_La_Liga = laLiga0919Filtered.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
y_La_Liga = laLiga0919Filtered['FTR'].copy()
scaler = MinMaxScaler()  # Center to the mean and component wise scale to unit variance.
X_La_Liga = scaler.fit_transform(X_La_Liga)
X_train, X_test, y_train, y_test = train_test_split(X_La_Liga, y_La_Liga, test_size=190, random_state=2, stratify=y_La_Liga)


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
clf_LogReg = LogisticRegression(random_state=1)
clf_SVC = SVC(random_state=2, kernel='rbf')
clf_xgb = xgb.XGBClassifier(seed=3)

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
parameters = {'learning_rate': [0.01, 0.1, 0.5],
              'n_estimators': [40, 60],
              'max_depth': [4, 5, 6],
              'min_child_weight': [3, 4, 5],
              'gamma': [0.4, 1],
              'subsample': [0.8, 1],
              'colsample_bytree': [0.8, 1],
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
print(grid_obj.best_params_)
print(clf)

## Reports:
# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


print('')
print('------------VALIDATION HAPPENING------------')
print('')


## Validation on HoldOut data:
X_La_Liga_HO = laLiga0919FilteredHoldOut.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1).copy()
scaler = MinMaxScaler()
X_La_Liga_HO = scaler.fit_transform(X_La_Liga_HO)
y_La_Liga_HO = laLiga0919FilteredHoldOut['FTR'].copy()
f1, acc = predict_labels(clf, X_La_Liga_HO, y_La_Liga_HO)
print(X_La_Liga_HO.shape, y_La_Liga_HO.shape)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))
