import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore

from Data_Cleaning_for_Analysis import la_Liga_Dict
from Data_Cleaning_for_Analysis import PL_Dict

warnings.filterwarnings("ignore")

plt.style.use(['seaborn-white', 'bmh'])


## Parameters:
anal_graphs_title_dict = {'Half Time Influence Graph': "Half-Time-Leader's result at Final-Time",
                          'Half Time Influence Graph (leader by 1)': "Half-Time-Leader's result at Final-Time (leads by exactly 1)",
                          'Half Time Influence Graph (leader by more than 2)': "Half-Time-Leader's result at Final-Time (leads by 2 or more)",
                          'Location Influence Graph': "Effect of Match Location on Winning"}

anal_graphs_leagues_names_dict = {'La Liga Name': 'La Liga', 'Premier League Name': 'Premier League'}

anal_graphs_xticklabels_dict = {'Half Time Influence Graph': np.array(["Leader Won", "Draw", "Leader Lost"]),
                                'Location Influence Graph': np.array(["Home Team Won", "Draw", "Away Team Won"])}

anal_graphs_xlabel_dict = {'Half Time Influence Graph': "Leader's Status",
                           'Location Influence Graph': "Winning teams/Draw"}


## Plots and Helpers:
# Bar plot's parameters of half time result's influence on final time result:
def HT_influence_bar_plot_param(league_df):
    num_of_games_that_have_lead_at_HT = len(league_df)
    num_of_games_that_leader_wins_at_FT = len(league_df[league_df['FTR'] == league_df['HTR']])
    num_of_games_that_draw_at_FT = len(league_df[league_df['FTR'] == 'D'])
    num_of_games_that_leader_loses_at_FT = num_of_games_that_have_lead_at_HT - \
                                           num_of_games_that_leader_wins_at_FT - \
                                           num_of_games_that_draw_at_FT

    percentages = [(num_of_games_that_leader_wins_at_FT / num_of_games_that_have_lead_at_HT) * 100,
                   (num_of_games_that_draw_at_FT / num_of_games_that_have_lead_at_HT) * 100,
                   (num_of_games_that_leader_loses_at_FT / num_of_games_that_have_lead_at_HT) * 100]

    return percentages


# Bar plot's parameters of game's whereabouts' influence on final time result:
def location_influence_bar_plot_param(league_df):
    league_df_1_Matches = len(league_df)
    H_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'H']) / league_df_1_Matches
    Draws_Percents_1 = len(league_df[league_df['FTR'] == 'D']) / league_df_1_Matches
    A_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'A']) / league_df_1_Matches

    percentages = [H_Wins_Percents_1 * 100,
                   Draws_Percents_1 * 100,
                   A_Wins_Percents_1 * 100]

    return percentages


# Foundations for general two teams bar plot comparison:
def two_leagues_bar_compare(league1_name, league2_name, title, xlabel, xticklabels, x_bar_param_league1, x_bar_param_league2, ylabel='Winning Chances'):
    ind = np.arange(3)  # the x locations for the groups
    width = 0.27  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, x_bar_param_league1, width, alpha=0.8, color='#07a64c')
    rects2 = ax.bar(ind + width * 1.05, x_bar_param_league2, width, alpha=0.8, color='#b7040e')

    ### Title, Labels, Legend
    ## Title
    plt.title(title)

    ## y label
    ax.set_ylabel(ylabel)

    ## x label
    plt.xlabel(xlabel)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(xticklabels)

    # plt.setp(ax.xaxis.get_majorticklabels()) # all kinds of parameters

    # Create offset transform by 5 points in x direction
    dx = -15 / 72.
    dy = 0 / 72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    # apply offset transform to all x ticklabels.
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)

    ## Legend
    ax.legend((rects1[0], rects2[0]), (league1_name, league2_name))
    plt.margins(y=0.1)
    plt.tight_layout()

    ## bar labels
    def autolabel(rects):
        for rect in rects:
            rectHeight = rect.get_height()
            rectWidth = rect.get_width()
            ax.text(rect.get_x() + rectWidth / 2., 1.005 * rectHeight, str('%g' % round(float(rectHeight), 2)) + '%', ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ## show
    plt.show()


## Function Calls:
two_leagues_bar_compare(anal_graphs_leagues_names_dict['La Liga Name'],
                        anal_graphs_leagues_names_dict['Premier League Name'],
                        anal_graphs_title_dict['Location Influence Graph'],
                        anal_graphs_xlabel_dict['Location Influence Graph'],
                        anal_graphs_xticklabels_dict['Location Influence Graph'],
                        location_influence_bar_plot_param(la_Liga_Dict['laLiga0919Filtered']),
                        location_influence_bar_plot_param(PL_Dict['PL0919Filtered']))

two_leagues_bar_compare(anal_graphs_leagues_names_dict['La Liga Name'],
                        anal_graphs_leagues_names_dict['Premier League Name'],
                        anal_graphs_title_dict['Half Time Influence Graph'],
                        anal_graphs_xlabel_dict['Half Time Influence Graph'],
                        anal_graphs_xticklabels_dict['Half Time Influence Graph'],
                        HT_influence_bar_plot_param(la_Liga_Dict['laLiga0919Filtered_no_Draws']),
                        HT_influence_bar_plot_param(PL_Dict['PL0919Filtered_no_Draws']))

two_leagues_bar_compare(anal_graphs_leagues_names_dict['La Liga Name'],
                        anal_graphs_leagues_names_dict['Premier League Name'],
                        anal_graphs_title_dict['Half Time Influence Graph (leader by 1)'],
                        anal_graphs_xlabel_dict['Half Time Influence Graph'],
                        anal_graphs_xticklabels_dict['Half Time Influence Graph'],
                        HT_influence_bar_plot_param(la_Liga_Dict['laLiga0919Filtered_leader_by1']),
                        HT_influence_bar_plot_param(PL_Dict['PL0919Filtered_leader_by1']))

two_leagues_bar_compare(anal_graphs_leagues_names_dict['La Liga Name'],
                        anal_graphs_leagues_names_dict['Premier League Name'],
                        anal_graphs_title_dict['Half Time Influence Graph (leader by more than 2)'],
                        anal_graphs_xlabel_dict['Half Time Influence Graph'],
                        anal_graphs_xticklabels_dict['Half Time Influence Graph'],
                        HT_influence_bar_plot_param(la_Liga_Dict['laLiga0919Filtered_leader_2']),
                        HT_influence_bar_plot_param(PL_Dict['PL0919Filtered_leader_2']))
