# ממוצע גולים למשחק עד המשחק הנתון. רק בהמשך העונה מתנרמל - האם להשאיר משחקים ראשונים?
#
# גולים מצטברים - מתחילת העונה בהכרח?

import pandas as pd
import sqlite3

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 16)
pd.set_option('display.max_rows', 250)


### Functions:
def df_creator(path, file):
    file = pd.read_csv(path + file)
    return file


def reset_index_df(dataframe):
    return dataframe.reset_index(drop=True, inplace=True)


def rename_leagues_columns(league_df, dictionary):
    league_df.rename(columns=dictionary, inplace=True)


def drop_first_rows(league_df, rows_to_drop=10):
    league_df = league_df.iloc[rows_to_drop:]

    return league_df


# Some functions create columns on specific teams, they are redundant. Function is needed at the end:
def drop_unnecessary_cols(seasons_matches):
    # Drop unnecessary columns:
    for col in seasons_matches.columns:
        for team in seasons_matches.groupby('HomeTeam').median().T.columns:
            if team in col:  # the moment we encounter a column that is specific for a team, we get rid of it
                seasons_matches = seasons_matches.drop([col, ], axis=1)
                break
    return seasons_matches


# Creates a column of teams' scored/conceded goals UNTIL current match:
def update_season_df_with_agg_goals_cols(season_matches):
    # New columns initialized.
    season_matches['HTAggGoalScoredMean'] = 0
    season_matches['HTAggGoalConcededMean'] = 0
    season_matches['ATAggGoalScoredMean'] = 0
    season_matches['ATAggGoalConcededMean'] = 0

    for team in season_matches.groupby('HomeTeam').median().T.columns:  # A way to iterate over teams
        # Iterated by team, now we mask original df for aggregating its goals. A new column is created for these values for each team. After loop,
        # we set these value to original df by making a new column as sum of all these columns. On the way we actually compute compute and insert their
        # means.
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


# give points according to result:
def league_points_setter(season_matches, teams_dict):
    num_of_matches = len(season_matches)

    for match_ind in range(num_of_matches):
        HomeTeam = season_matches.iloc[match_ind]['HomeTeam']  # Home Team of current match
        AwayTeam = season_matches.iloc[match_ind]['AwayTeam']

        if season_matches.iloc[match_ind]['FTR'] == 'H':
            teams_dict[HomeTeam].append(3)
            teams_dict[AwayTeam].append(0)
        elif season_matches.iloc[match_ind]['FTR'] == 'A':
            teams_dict[HomeTeam].append(0)
            teams_dict[AwayTeam].append(3)
        else:
            teams_dict[HomeTeam].append(1)
            teams_dict[AwayTeam].append(1)


# Returns a df with teams' agg league points:
def get_agg_points(season_matches):
    # Create a dictionary with team names as keys
    teams = {}
    for team in season_matches.groupby('HomeTeam').median().T.columns:
        teams[team] = []

    league_points_setter(season_matches, teams)

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
def update_concat_df_with_last_3_specific_FTRs_cols1(seasons_matches):  # Notice that applies for CONCATENATED df
    # New columns initialized and remapping dictionaries initialized.
    seasons_matches['HTLast3SpecificMatches'] = 0
    HomeTeam_dict = {'H': 1, 'D': 0, 'A': 0}

    seasons_matches['ATLast3SpecificMatches'] = 0
    AwayTeam_dict = {'A': 1, 'D': 0, 'H': 0}

    for team in seasons_matches.groupby('HomeTeam').median().T.columns:  # A way to iterate over teams
        # Iterated by team, now we mask original df for aggregating league points. A new column is created for these values for each team. After loop,
        # we set these value to original df by making a new column as sum of all these columns. On the way we actually compute and insert their means.
        season_matches_of_team_when_home = seasons_matches[seasons_matches['HomeTeam'] == team].copy()
        season_matches_of_team_when_home[team + 'AsHTSpecPastResults'] = season_matches_of_team_when_home['FTR']  # Duplicating column to remap it.
        season_matches_of_team_when_home[team + 'AsHTSpecPastResults'] = season_matches_of_team_when_home[team + 'AsHTSpecPastResults'].map(HomeTeam_dict)
        seasons_matches[team + 'AsHTSpec3PastResults'] = season_matches_of_team_when_home[team + 'AsHTSpecPastResults'].rolling(3).sum()

        season_matches_of_team_when_away = seasons_matches[seasons_matches['AwayTeam'] == team].copy()
        season_matches_of_team_when_away[team + 'AsATSpecPastResults'] = season_matches_of_team_when_away['FTR']  # Duplicating column to remap it.
        season_matches_of_team_when_away[team + 'AsATSpecPastResults'] = season_matches_of_team_when_away[team + 'AsATSpecPastResults'].map(AwayTeam_dict)
        seasons_matches[team + 'AsATSpec3PastResults'] = season_matches_of_team_when_away[team + 'AsATSpecPastResults'].rolling(3).sum()

    seasons_matches.fillna(0, inplace=True)

    for col in seasons_matches.columns:
        if 'AsHTSpec3PastResults' in col:
            seasons_matches['HTLast3SpecificMatches'] += seasons_matches[col]
        elif 'AsATSpec3PastResults' in col:
            seasons_matches['ATLast3SpecificMatches'] += seasons_matches[col]

    return seasons_matches


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
def update_concat_df_with_last_3_any_FTRs_cols1(seasons_matches):  # Notice that applies for CONCATENATED df
    # New columns initialized and remapping dictionaries initialized.
    seasons_matches['HTLast3AnyMatches'] = 0
    HomeTeam_dict = {'H': 1, 'D': 0, 'A': 0}

    seasons_matches['ATLast3AnyMatches'] = 0
    AwayTeam_dict = {'A': 1, 'D': 0, 'H': 0}

    for team in seasons_matches.groupby('HomeTeam').median().T.columns:  # A way to iterate over teams
        # Iterated by team, now we mask original df for aggregating league points. A new column is created for these values for each team. After loop,
        # we set these value to original df by making a new column as sum of all these columns. On the way we actually compute and insert their means.
        season_matches_of_team_when_home = seasons_matches[seasons_matches['HomeTeam'] == team].copy()
        season_matches_of_team_when_home[team + 'AsHTSpecPastResults'] = season_matches_of_team_when_home['FTR']  # Duplicating column to remap it.
        season_matches_of_team_when_home[team + 'AsHTSpecPastResults'] = season_matches_of_team_when_home[team + 'AsHTSpecPastResults'].map(HomeTeam_dict)
        seasons_matches[team + 'AsHTSpec3PastResults'] = season_matches_of_team_when_home[team + 'AsHTSpecPastResults'].rolling(3).sum()

        season_matches_of_team_when_away = seasons_matches[seasons_matches['AwayTeam'] == team].copy()
        season_matches_of_team_when_away[team + 'AsATSpecPastResults'] = season_matches_of_team_when_away['FTR']  # Duplicating column to remap it.
        season_matches_of_team_when_away[team + 'AsATSpecPastResults'] = season_matches_of_team_when_away[team + 'AsATSpecPastResults'].map(AwayTeam_dict)
        seasons_matches[team + 'AsATSpec3PastResults'] = season_matches_of_team_when_away[team + 'AsATSpecPastResults'].rolling(3).sum()

    seasons_matches.fillna(0, inplace=True)

    for col in seasons_matches.columns:
        if 'AsHTSpec3PastResults' in col:
            seasons_matches['HTLast3SpecificMatches'] += seasons_matches[col]
        elif 'AsATSpec3PastResults' in col:
            seasons_matches['ATLast3SpecificMatches'] += seasons_matches[col]

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
def update_concat_df_with_team_location_influence(seasons_matches):
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

# # Update DFs with new relevant data (not on concatenated since it is per league)
# for la_Liga_season in experiment_list:
#     update_season_df_with_agg_goals_cols(la_Liga_season)
#     update_season_df_with_teams_points_col(la_Liga_season)
#
# # Get rid of non informative rows:
# for la_Liga_season in experiment_list:
#     la_Liga_season = drop_first_rows(la_Liga_season)

laLiga0919FilteredML = pd.concat(file for file in experiment_list)
# laLiga0919FilteredML = pd.concat(file for file in laLigaSeasonsFilteredList)
reset_index_df(laLiga0919FilteredML)
update_concat_df_with_last_3_specific_FTRs_cols1(laLiga0919FilteredML)
# update_concat_df_with_last_3_any_FTRs_cols(laLiga0919FilteredML)
# update_concat_df_with_team_location_influence(laLiga0919FilteredML)
# laLiga0919FilteredML = drop_unnecessary_cols(laLiga0919FilteredML)
print(laLiga0919FilteredML.head(250))
print(laLiga0919FilteredML.columns)
