import pandas as pd
import sqlite3

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)


## Parameters:
relevant_anal_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']

laLiga_cols_renaming = {'HS': 'Home Shots', 'AS': 'Away Shots', 'HST': 'Home Shots on Target', 'AST': 'Away Shots on Target',
                        'HF': 'Home Fouls Committed', 'AF': 'Away Fouls Committed', 'HC': 'Home Corners', 'AC': 'Away Corners',
                        'HY': 'Home Yellows', 'AY': 'Away Yellows', 'HR': 'Home Reds', 'AR': 'Away Reds'}


## Functions:
def df_creator(path, file):
    file = pd.read_csv(path + file)
    return file


def abs_goal_difference_calculator(df_league):
    return abs(pd.to_numeric(df_league['HTHG']) - pd.to_numeric(df_league['HTAG']))


def reset_index_df(dataframe):
    return dataframe.reset_index(drop=True, inplace=True)


def rename_leagues_columns(league_df, dictionary):
    league_df.rename(columns=dictionary, inplace=True)


# Modify a league df's attributes - relevant columns, relevant games (by score):
def modify_league(league_df, relevant_col_list, with_draws=True, exact_diff=False, diff_at_least=False, lead_diff=None, start_ind=0):
    league_df_modified = league_df[start_ind:].copy()

    league_df_modified = league_df_modified[relevant_col_list]

    if not with_draws:
        league_df_modified = league_df_modified[league_df_modified['HTR'] != 'D']

    if exact_diff:
        league_df_modified = league_df_modified[abs_goal_difference_calculator(league_df_modified) == lead_diff]
    elif diff_at_least:
        league_df_modified = league_df_modified[abs_goal_difference_calculator(league_df_modified) >= lead_diff]

    return league_df_modified


## Reading the La Liga data files and concatenate the DFs:
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

## Master Premier League ("PL") df extracted:
con = sqlite3.connect("C:/Users/User/PycharmProjects/Football-Data-Analysis/EPL_Seasons_1993-2017_RAW_Table.sqlite")
PL_raw_table = pd.read_sql_query("SELECT * FROM EPL", con)


## Modifying the La Liga and PL dataframes:
rename_leagues_columns(laLiga0919Concat, laLiga_cols_renaming)
rename_leagues_columns(PL_raw_table, laLiga_cols_renaming)
la_Liga_Dict = {'laLiga0919Filtered': modify_league(laLiga0919Concat, relevant_anal_cols),
                'laLiga0919Filtered_no_Draws': modify_league(laLiga0919Concat, relevant_anal_cols, with_draws=False),
                'laLiga0919Filtered_leader_by1': modify_league(laLiga0919Concat, relevant_anal_cols, with_draws=False, exact_diff=True, lead_diff=1),
                'laLiga0919Filtered_leader_2': modify_league(laLiga0919Concat, relevant_anal_cols, with_draws=False, diff_at_least=True, lead_diff=2)}
PL_Dict = {'PL0919Filtered': modify_league(PL_raw_table, relevant_anal_cols),
           'PL0919Filtered_no_Draws': modify_league(PL_raw_table, relevant_anal_cols, with_draws=False, start_ind=924),
           'PL0919Filtered_leader_by1': modify_league(PL_raw_table, relevant_anal_cols, with_draws=False, exact_diff=True, lead_diff=1, start_ind=924),
           'PL0919Filtered_leader_2': modify_league(PL_raw_table, relevant_anal_cols, with_draws=False, diff_at_least=True, lead_diff=2, start_ind=924)}

for key, df in PL_Dict.items():
    reset_index_df(df)
