import pandas as pd
from pandas import to_numeric
import sqlite3

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 15)


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

### Function Helpers for both stages:
def df_creator(path, file):
    file = pd.read_csv(path + file)
    return file


def abs_goal_diff_calc(df_league):
    return abs(to_numeric(df_league['HTHG']) - to_numeric(df_league['HTAG']))


def reset_index_df(dataframe):
    return dataframe.reset_index(drop=True, inplace=True)


def rename_leagues_columns(league_df, dictionary):
    league_df.rename(columns=dictionary, inplace=True)


# Returns a dataframe of the agg goals scored, arranged by teams (rows) and matchweek (columns):
def get_agg_goals_scored(season_matches):
    num_of_matches = len(season_matches)
    # Create a dictionary with team names as keys
    teams = {}
    for team in season_matches.groupby('HomeTeam').median().T.columns:  # A way to turn the teams into the columns
        teams[team] = []

    for match_ind in range(num_of_matches):
        HomeTeamGoalsScored = season_matches.iloc[match_ind]['FTHG']
        AwayTeamGoalsScored = season_matches.iloc[match_ind]['FTAG']
        teams[season_matches.iloc[match_ind]['HomeTeam']].append(HomeTeamGoalsScored)  # Inserts on "teams" dictionary the team's goals
        teams[season_matches.iloc[match_ind]['AwayTeam']].append(AwayTeamGoalsScored)  # Inserts on "teams" dictionary the team's goals

    # Create a dataframe for goals scored where rows are teams and cols are the matchweek's goals for the team. list breaks into columns.
    GoalsScoredByTeam = pd.DataFrame(data=teams, index=[index for index in range(1, 39)]).T  # Teams are rows again.
    GoalsScoredByTeam[0] = 0  # This will become the aggregate goals scored
    # Aggregate to get until that point (df values turn into cumulative sum of former values):
    for match_week in range(2, 39):
        GoalsScoredByTeam[match_week] = GoalsScoredByTeam[match_week] + GoalsScoredByTeam[match_week - 1]

    return GoalsScoredByTeam


# Returns a dataframe of the agg goals conceded, arranged by teams (rows) and matchweek (columns), see former function for documentation:
def get_agg_goals_conceded(season_matches):
    num_of_matches = len(season_matches)
    teams = {}
    for team in season_matches.groupby('HomeTeam').median().T.columns:
        teams[team] = []

    for match_ind in range(num_of_matches):
        HomeTeamGoalsConceded = season_matches.iloc[match_ind]['FTAG']  # (There's no mistake here)
        AwayTeamGoalsConceded = season_matches.iloc[match_ind]['FTHG']
        teams[season_matches.iloc[match_ind]['HomeTeam']].append(HomeTeamGoalsConceded)
        teams[season_matches.iloc[match_ind]['AwayTeam']].append(AwayTeamGoalsConceded)

    GoalsConcededByTeam = pd.DataFrame(data=teams, index=[index for index in range(1, 39)]).T
    GoalsConcededByTeam[0] = 0  # Is that code line necessary
    for match_week in range(2, 39):
        GoalsConcededByTeam[match_week] = GoalsConcededByTeam[match_week] + GoalsConcededByTeam[match_week - 1]

    return GoalsConcededByTeam


def update_season_matches_with_agg_goals(season_matches):
    num_of_matches = len(season_matches)
    agg_goals_scored = get_agg_goals_scored(season_matches)
    agg_goals_conceded = get_agg_goals_conceded(season_matches)

    match_week = 0
    HTAggGoalScored = []
    ATAggGoalScored = []
    HTAggGoalConceded = []
    ATAggGoalConceded = []

    # Updates the lists in accordance with
    for match_ind in range(num_of_matches):
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']  # Away Team of current match
        HTAggGoalScored.append(agg_goals_scored.loc[HT][match_week])  # Appends value of agg scored goals of HT as of current match
        ATAggGoalScored.append(agg_goals_scored.loc[AT][match_week])  # Appends value of agg scored goals of AT as of current match
        HTAggGoalConceded.append(agg_goals_conceded.loc[HT][match_week])  # Appends value of agg conceded goals of HT as of current match
        ATAggGoalConceded.append(agg_goals_conceded.loc[AT][match_week])  # Appends value of agg conceded goals of AT as of current match

        # We move to next week=column after 10 matches have been played (notice that a match accounts for 2 teams and we have 20 rows=teams
        # overall in the AggGoal Dataframes). Meaning one value of match_week sweeps through all teams:
        if ((match_ind + 1) % 10) == 0:
            match_week = match_week + 1

    # Updates the season_matches df by creating new columns according to above lists:
    season_matches['HomeTeamAggGoalScored'] = HTAggGoalScored
    season_matches['AwayTeamAggGoalScored'] = ATAggGoalScored
    season_matches['HomeTeamAggGoalConceded'] = HTAggGoalConceded
    season_matches['AwayTeamAggGoalConceded'] = ATAggGoalConceded

    return season_matches


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
laLiga_cols_renaming = {'HS': 'Home Shots', 'AS': 'Away Shots', 'HST': 'Home Shots on Target', 'AST': 'Away Shots on Target',
                        'HF': 'Home Fouls Committed', 'AF': 'Away Fouls Committed', 'HC': 'Home Corners', 'AC': 'Away Corners',
                        'HY': 'Home Yellows', 'AY': 'Away Yellows', 'HR': 'Home Reds', 'AR': 'Away Reds'}
rename_leagues_columns(laLiga0919Concat, laLiga_cols_renaming)
relevant_analysis_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
relevant_ML_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

### Master Premier League df extracted:
con = sqlite3.connect("C:/Users/User/PycharmProjects/Football-Data-Analysis/EPL_Seasons_1993-2017_RAW_Table.sqlite")
dfRawTable = pd.read_sql_query("SELECT * FROM EPL", con)

### Data Analysis Stage:
## Modifying the La Liga DF:
# Leave relevant columns:
laLiga0919Filtered = laLiga0919Concat[relevant_analysis_cols].copy()
# Filter out games that draw at HT:
laLiga0919Filtered2 = laLiga0919Filtered[((laLiga0919Filtered['HTR'] == 'H') | (laLiga0919Filtered['HTR'] == 'A'))].copy()  # No Draws
# Filter out games that draw at HT and leader leads by exactly 1:
laLiga0919Filtered3 = laLiga0919Filtered2[abs_goal_diff_calc(laLiga0919Filtered2) == 1].copy()  # Leader leads by exactly 1
# Filter out games that draw at HT and leader leads by more than 1:
laLiga0919Filtered4 = laLiga0919Filtered2[abs_goal_diff_calc(laLiga0919Filtered2) > 1].copy()  # Leader leads by more than 1

## Modifying the Premier League DF:
# Leave relevant columns:
premierLeague9518Filtered = dfRawTable[924:][relevant_analysis_cols].copy()
# Filter out games that draw at HT:
premierLeague9518Filtered2 = premierLeague9518Filtered[((premierLeague9518Filtered['HTR'] == 'H') |
                                                        (premierLeague9518Filtered['HTR'] == 'A'))].copy()
# Filter out games that draw at HT and leader leads by exactly 1:
premierLeague9518Filtered3 = premierLeague9518Filtered2[abs_goal_diff_calc(premierLeague9518Filtered2) == 1].copy()
# Filter out games that draw at HT and leader leads by more than 1:
premierLeague9518Filtered4 = premierLeague9518Filtered2[abs_goal_diff_calc(premierLeague9518Filtered2) > 1].copy()

reset_index_list = [premierLeague9518Filtered2, premierLeague9518Filtered3, premierLeague9518Filtered4]
for df in reset_index_list:
    reset_index_df(df)

#### ML Stage:
### La Liga df modification:
la_liga_season_0910_filtered_ML = df_creator(la_liga_path, 'season-0910_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1011_filtered_ML = df_creator(la_liga_path, 'season-1011_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1112_filtered_ML = df_creator(la_liga_path, 'season-1112_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1213_filtered_ML = df_creator(la_liga_path, 'season-1213_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1314_filtered_ML = df_creator(la_liga_path, 'season-1314_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1415_filtered_ML = df_creator(la_liga_path, 'season-1415_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1516_filtered_ML = df_creator(la_liga_path, 'season-1516_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1617_filtered_ML = df_creator(la_liga_path, 'season-1617_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1718_filtered_ML = df_creator(la_liga_path, 'season-1718_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1819_filtered_ML = df_creator(la_liga_path, 'season-1819_csv.csv')[relevant_ML_cols].copy()
laLigaLeaguesFilteredList = [la_liga_season_0910_filtered_ML,
                             la_liga_season_1011_filtered_ML,
                             la_liga_season_1112_filtered_ML,
                             la_liga_season_1213_filtered_ML,
                             la_liga_season_1314_filtered_ML,
                             la_liga_season_1415_filtered_ML,
                             la_liga_season_1516_filtered_ML,
                             la_liga_season_1617_filtered_ML,
                             la_liga_season_1718_filtered_ML,
                             la_liga_season_1819_filtered_ML]

laLiga0919FilteredML = pd.concat(file for file in laLigaLeaguesFilteredList)
# print(laLiga0919FilteredML.columns)

X_La_Liga = laLiga0919FilteredML.drop(['FTR'], axis=1)
# print(X_La_Liga.head())
y_La_Liga = laLiga0919FilteredML['FTR']
# print(y_La_Liga.head())

update_season_matches_with_agg_goals(la_liga_season_0910_filtered_ML)