import pandas as pd
import sqlite3

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

### notes on files: http://www.football-data.co.uk/notes.txt:
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


### Functions:
def df_creator(path, file):
    file = pd.read_csv(path + file)
    return file


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
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']
        HomeTeamGoalsScored = season_matches.iloc[match_ind]['FTHG']
        AwayTeamGoalsScored = season_matches.iloc[match_ind]['FTAG']
        teams[HT].append(HomeTeamGoalsScored)  # Inserts on "teams" dictionary the team's goals.
        teams[AT].append(AwayTeamGoalsScored)  # Inserts on "teams" dictionary the team's goals.

    # Create a dataframe for goals scored where rows are teams and cols are the matchweek's goals for the team. list breaks into columns:
    GoalsScoredByTeam = pd.DataFrame(data=teams, index=[index for index in range(1, 39)]).T  # Teams are rows again.
    GoalsScoredByTeam[0] = 0  # Because before week 1, 0 goals were scored (current game always excluded since it is unknown yet). it is
    # relevant to function "update_season_matches_df_with_agg_goals_cols" later
    # Aggregates to get scored goals UNTIL current game (excludes current since it is unknown yet), df values
    # turn into cumulative sum of former values:
    for match_week in range(2, 39):  # (Remember that actually the first relevant GoalsScoredByTeam column=match_week is 1 and not 0)
        GoalsScoredByTeam[match_week] = GoalsScoredByTeam[match_week] + GoalsScoredByTeam[match_week - 1]

    return GoalsScoredByTeam


# Returns a dataframe of the agg goals conceded, arranged by teams (rows) and matchweek (columns), see agg scored func for documentations:
def get_agg_goals_conceded(season_matches):
    num_of_matches = len(season_matches)
    teams = {}
    for team in season_matches.groupby('HomeTeam').median().T.columns:
        teams[team] = []

    for match_ind in range(num_of_matches):
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']
        HomeTeamGoalsConceded = season_matches.iloc[match_ind]['FTAG']  # (There's no mistake here of course)
        AwayTeamGoalsConceded = season_matches.iloc[match_ind]['FTHG']
        teams[HT].append(HomeTeamGoalsConceded)
        teams[AT].append(AwayTeamGoalsConceded)

    GoalsConcededByTeam = pd.DataFrame(data=teams, index=[index for index in range(1, 39)]).T
    GoalsConcededByTeam[0] = 0
    for match_week in range(2, 39):
        GoalsConcededByTeam[match_week] = GoalsConcededByTeam[match_week] + GoalsConcededByTeam[match_week - 1]

    return GoalsConcededByTeam


# Creates a column of teams' scored/conceded goals UNTIL current match:
def update_season_df_with_agg_goals_cols(season_matches):
    num_of_matches = len(season_matches)
    agg_goals_scored = get_agg_goals_scored(season_matches)
    agg_goals_conceded = get_agg_goals_conceded(season_matches)

    match_week = 0
    HTAggGoalScored = []
    ATAggGoalScored = []
    HTAggGoalConceded = []
    ATAggGoalConceded = []

    # Updates the lists with agg goals in accordance with matchweek:
    for match_ind in range(num_of_matches):
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']  # Away Team of current match
        HTAggGoalScored.append(agg_goals_scored.loc[HT][match_week])  # Appends num of agg scored goals of HT UNTIL current match
        ATAggGoalScored.append(agg_goals_scored.loc[AT][match_week])  # Appends num of agg scored goals of AT UNTIL current match
        HTAggGoalConceded.append(agg_goals_conceded.loc[HT][match_week])  # Appends num of agg conceded goals of HT UNTIL current match
        ATAggGoalConceded.append(agg_goals_conceded.loc[AT][match_week])  # Appends num of agg conceded goals of AT UNTIL current match

        # We move to next week=column after 10 matches have been played (notice that a match accounts for 2 teams and we have 20 rows=teams
        # overall in the AggGoal Dataframes). Meaning one value of match_week sweeps through all teams:
        if ((match_ind + 1) % 10) == 0:
            match_week = match_week + 1

    # Updates the season_matches df by creating new columns according to above lists:
    season_matches['HTAggGoalScored'] = HTAggGoalScored
    season_matches['ATAggGoalScored'] = ATAggGoalScored
    season_matches['HTAggGoalConceded'] = HTAggGoalConceded
    season_matches['ATAggGoalConceded'] = ATAggGoalConceded

    return season_matches


# Returns a df with teams' agg league points:
def get_agg_points(season_matches):
    num_of_matches = len(season_matches)

    # Create a dictionary with team names as keys
    teams = {}
    for team in season_matches.groupby('HomeTeam').median().T.columns:
        teams[team] = []

    # Fill the dictionary values (lists) with league points:
    for match_ind in range(num_of_matches):
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']

        if season_matches.iloc[match_ind]['FTR'] == 'H':
            teams[HT].append(3)
            teams[AT].append(0)
        elif season_matches.iloc[match_ind]['FTR'] == 'A':
            teams[HT].append(0)
            teams[AT].append(3)
        else:
            teams[HT].append(1)
            teams[AT].append(1)

    teams_league_points_df = pd.DataFrame(data=teams, index=[index for index in range(1, 39)]).T
    teams_league_points_df[0] = 0
    # Aggregates to get league points UNTIL current game (excludes current since it is unknown yet), df values
    # turn into cumulative sum of former values:
    for match_week in range(2, 39):
        teams_league_points_df[match_week] = teams_league_points_df[match_week] + teams_league_points_df[match_week - 1]

    return teams_league_points_df


# Creates a column of teams' points UNTIL current match:
def update_season_df_with_teams_points_col(season_matches):
    teams_league_points_df = get_agg_points(season_matches)
    num_of_matches = len(season_matches)
    match_week = 0
    HTAggLeaguePoints = []
    ATAggLeaguePoints = []

    # Updates the lists with agg goals in accordance with matchweek:
    for match_ind in range(num_of_matches):
        HT = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AT = season_matches.iloc[match_ind]['AwayTeam']  # Away Team of current match
        HTAggLeaguePoints.append(teams_league_points_df.loc[HT][match_week])  # Appends num of agg league points of HT UNTIL current match
        ATAggLeaguePoints.append(teams_league_points_df.loc[AT][match_week])  # Appends num of agg league points of AT UNTIL current match

        # We move to next week=column after 10 matches have been played (notice that a match accounts for 2 teams and we have 20 rows=teams
        # overall in the AggGoal Dataframes). Meaning one value of match_week sweeps through all teams:
        if ((match_ind + 1) % 10) == 0:
            match_week = match_week + 1

    # Updates the season_matches df by creating new columns according to above lists:
    season_matches['HTAggLeaguePoints'] = HTAggLeaguePoints
    season_matches['ATAggLeaguePoints'] = ATAggLeaguePoints

    return season_matches


# Creates a column of teams' number of wins on last 3 matches between the two:
def update_concat_df_with_last_3_specific_FTRs_cols(seasons_matches):  # Notice that applies for CONCATENATED df
    num_of_matches = len(seasons_matches)
    seasons_matches['NumOfPastHTSpecificWinsOutOf3'] = 0
    seasons_matches['NumOfPastATSpecificWinsOutOf3'] = 0

    # Compute the relevant values and set them in the season_matches df:
    for general_match_ind in range(0, num_of_matches):
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
def update_concat_df_with_last_3_any_FTRs_cols(seasons_matches):
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
                if (HT == HT_past_match and FTR_past_match == 'H') or (HT == AT_past_match and FTR_past_match == 'A'):
                    HT_win_count = HT_win_count + 1
                HT_history_monitor = HT_history_monitor + 1
            if AT in [HT_past_match, AT_past_match]:  # Not elif because theoretically last game could be between both
                if (AT == HT_past_match and FTR_past_match == 'H') or (AT == AT_past_match and FTR_past_match == 'A'):
                    AT_win_count = AT_win_count + 1
                AT_history_monitor = AT_history_monitor + 1

            if HT_history_monitor == 3 and AT_history_monitor == 3:
                break  # Stop when 3 games were taken into account or before that when we reached beginning of data

        seasons_matches.at[general_match_ind, 'NumOfPastHTWinsOutOfLast3Matches'] = HT_win_count  # resets value in df
        seasons_matches.at[general_match_ind, 'NumOfPastATWinsOutOfLast3Matches'] = AT_win_count

    return seasons_matches


# Creates columns of match's whereabouts' influence on FTR. If team x is at home then HTWinningChancesAtHome column's value is its num of games won
# at home divided by num of its games played at home. Similarly for that HT team, we have columns HTLosingChancesAtHome and HTDrawChancesAtHome
# function update_concat_df_with_percent_of_wins_by_location will be redundant.
def update_concat_df_with_team_location_influence(seasons_matches):
    seasons_matches['HTWinningChancesAtHome'] = 0
    seasons_matches['HTLosingChancesAtHome'] = 0
    seasons_matches['HTDrawChancesAtHome'] = 0
    groupByHTdf = seasons_matches.groupby('HomeTeam')
    for key, item in groupByHTdf:
        numOfHTGamesPlayedAtHome = len(item)
        numOfHTGamesWonAtHome = item[item['FTR'] == 'H'].shape[0]
        numOfHTGamesLostAtHome = item[item['FTR'] == 'A'].shape[0]
        numOfHTGamesDrawnAtHome = item[item['FTR'] == 'D'].shape[0]
        seasons_matches.loc[seasons_matches['HomeTeam'] == key, 'HTWinningChancesAtHome'] = numOfHTGamesWonAtHome/numOfHTGamesPlayedAtHome
        seasons_matches.loc[seasons_matches['HomeTeam'] == key, 'HTLosingChancesAtHome'] = numOfHTGamesLostAtHome / numOfHTGamesPlayedAtHome
        seasons_matches.loc[seasons_matches['HomeTeam'] == key, 'HTLosingChancesAtHome'] = numOfHTGamesDrawnAtHome / numOfHTGamesPlayedAtHome

    seasons_matches['ATWinningChancesWhenAway'] = 0
    seasons_matches['ATLosingChancesWhenAway'] = 0
    seasons_matches['ATDrawChancesWhenAway'] = 0
    groupByATdf = seasons_matches.groupby('AwayTeam')
    for key, item in groupByATdf:
        numOfATGamesPlayedWhenAway = len(item)
        numOfATGamesWonWhenAway = item[item['FTR'] == 'A'].shape[0]
        seasons_matches.loc[seasons_matches['AwayTeam'] == key, 'ATWinningChancesWhenAway'] = numOfATGamesWonWhenAway / numOfATGamesPlayedWhenAway

    return seasons_matches


### Reading the La Liga data files and concatenate the DFs:
la_liga_path = "C:/Users/User/PycharmProjects/Football-Data-Analysis/"
relevant_ML_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

### Master Premier League df extracted:
con = sqlite3.connect("C:/Users/User/PycharmProjects/Football-Data-Analysis/EPL_Seasons_1993-2017_RAW_Table.sqlite")
dfRawTable = pd.read_sql_query("SELECT * FROM EPL", con)


### La Liga df modification:
la_liga_season_0910_filtered_ML = df_creator(la_liga_path, 'season-0910_csv.csv')[relevant_ML_cols].copy()  # Every file separately because some
                                                                                                            # functions are per league.
la_liga_season_1011_filtered_ML = df_creator(la_liga_path, 'season-1011_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1112_filtered_ML = df_creator(la_liga_path, 'season-1112_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1213_filtered_ML = df_creator(la_liga_path, 'season-1213_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1314_filtered_ML = df_creator(la_liga_path, 'season-1314_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1415_filtered_ML = df_creator(la_liga_path, 'season-1415_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1516_filtered_ML = df_creator(la_liga_path, 'season-1516_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1617_filtered_ML = df_creator(la_liga_path, 'season-1617_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1718_filtered_ML = df_creator(la_liga_path, 'season-1718_csv.csv')[relevant_ML_cols].copy()
la_liga_season_1819_filtered_ML = df_creator(la_liga_path, 'season-1819_csv.csv')[relevant_ML_cols].copy()
laLigaSeasonsFilteredList = [la_liga_season_0910_filtered_ML,
                             la_liga_season_1011_filtered_ML,
                             la_liga_season_1112_filtered_ML,
                             la_liga_season_1213_filtered_ML,
                             la_liga_season_1314_filtered_ML,
                             la_liga_season_1415_filtered_ML,
                             la_liga_season_1516_filtered_ML,
                             la_liga_season_1617_filtered_ML,
                             la_liga_season_1718_filtered_ML,
                             la_liga_season_1819_filtered_ML]
experiment_list = [la_liga_season_0910_filtered_ML]

# Update DFs with new relevant data (not on concatenated since it is per league)
for la_Liga_season in laLigaSeasonsFilteredList:
    update_season_df_with_agg_goals_cols(la_Liga_season)
    update_season_df_with_teams_points_col(la_Liga_season)

# laLiga0919FilteredML = pd.concat(file for file in laLigaSeasonsFilteredList)
laLiga0919FilteredML = pd.concat(file for file in experiment_list)
reset_index_df(laLiga0919FilteredML)
# update_concat_df_with_last_3_specific_FTRs_cols(laLiga0919FilteredML)
# update_concat_df_with_last_3_any_FTRs_cols(laLiga0919FilteredML)
# update_concat_df_with_percent_of_wins_by_location(laLiga0919FilteredML)
update_concat_df_with_team_location_influence(laLiga0919FilteredML)
print(laLiga0919FilteredML.head())
print(laLiga0919FilteredML.columns)


X_La_Liga = laLiga0919FilteredML.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1)
# print(X_La_Liga.head())
y_La_Liga = laLiga0919FilteredML['FTR']
# print(y_La_Liga.head())

# print(get_agg_goals_scored(la_liga_season_0910_filtered_ML).head(12))
# print(get_agg_goals_conceded(la_liga_season_0910_filtered_ML).head(15))
# print(la_liga_season_0910_filtered_ML.head(12))
# print(get_agg_points(la_liga_season_0910_filtered_ML))
# print(update_season_matches_df_with_teams_points_col(la_liga_season_0910_filtered_ML))
