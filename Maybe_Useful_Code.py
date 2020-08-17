#### Leagues_Data_Cleaning:
### Helper Functions:
# # Collects aggregate score difference from last 3 games between the two team
# def get_last_3_games_aggregate_for_specific_teams(league_df, team1, team2, league_df_match_index):
#     HomeTeam = league_df['HomeTeam']
#     AwayTeam = league_df['AwayTeam']
#     league_df['TeamsAggDiff'] = None
#     relevant_teams_reduction_df = league_df.loc[team1 in league_df['TeamsList'] and team2 in league_df['TeamsList']].copy()
#     last_game_score = relevant_teams_reduction_df['FTHG'].iloc[-1] -
#     counter = 0
#     match_index = league_df_match_index - 1
#
#     while counter < 3:
#         league_df.iloc
#     for match_index in range(0, league_df_match_index): # runs through league games until current game and not after
#         if team1 in league_df['TeamsList'] and team2 in league_df['TeamsList']: # finds a former match of the two teams
#             if
#
# def get_total_goals_scored_in_season_until_match(season_league_df, team):
#     total_goals_scored_in_season_until_match = 0
#     # for match in season_league_df:
#     # total_goals_scored_in_season_until_match +=
#     return (team,)
#
#
# # Collects aggregate score difference from last 3 games between the two team
# def get_last_3_games_aggregate_for_specific_teams(league_df, team1, team2, league_df_match_index):
#     HomeTeam = league_df['HomeTeam']
#     AwayTeam = league_df['AwayTeam']
#     league_df['TeamsAggDiff'] = None
#     relevant_teams_reduction_df = league_df.loc[team1 in league_df['TeamsList'] and team2 in league_df['TeamsList']].copy()
#     # last_game_score = relevant_teams_reduction_df['FTHG'].iloc[-1] -
#     match_index = league_df_match_index - 1


### Analysis Stage:
# la_liga_0919_df['Year'] = pd.DatetimeIndex(la_liga_0919_df['Date']).year  # year column.
# premierLeague9518Filtered['Year'] = pd.DatetimeIndex(la_liga_0919_df['Date']).year  # year column.

### ML Stage:
# Home_Away_Merger(laLiga0919FilteredML)

# # Parameters of game's whereabouts' influence on final time result:
# def location_influence_bar_plot_param(league_df):
#     league_df_1_Matches = len(league_df)
#     H_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'H']) / league_df_1_Matches
#     Draws_Percents_1 = len(league_df[league_df['FTR'] == 'D']) / league_df_1_Matches
#     A_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'A']) / league_df_1_Matches
#
#     percentages = [H_Wins_Percents_1 * 100,
#                    Draws_Percents_1 * 100,
#                    A_Wins_Percents_1 * 100]
#     # percentages not multiplied by 100 so to keep them in the same area as other values (even if after we scale)
#
#     return percentages
#
#
# # Creates a column of match's whereabouts' influence on FTR (probability of winning)
# def update_concat_df_with_percent_of_wins_by_location(seasons_matches):
#     percentages_list = location_influence_bar_plot_param(seasons_matches)
#     seasons_matches['HT %'] = percentages_list[0]
#     seasons_matches['Draw %'] = percentages_list[1]
#     seasons_matches['AT %'] = percentages_list[2]
#     # (Allegedly function gives Data Leakage, though the assumption is that these are true more or less - always (at least for the past 20 years))
#
#     return seasons_matches


#### Analysis_Functions:



#### ML_Functions
