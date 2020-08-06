import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
# import xgboost as xgb

from Leagues_Data_and_Adaptations import laLiga0919Filtered2, laLiga0919Filtered3, laLiga0919Filtered4
from Leagues_Data_and_Adaptations import premierLeague9518Filtered2, premierLeague9518Filtered3, premierLeague9518Filtered4
from Leagues_Data_and_Adaptations import laLiga0919FilteredML, premierLeague9518FilteredML

plt.style.use(['seaborn-white', 'bmh'])

Analysis_graph1_title = "Half-Time-Leader's result at Final-Time"
Analysis_graph2_title = "Half-Time-Leader's result at Final-Time (leads by exactly 1)"
Analysis_graph3_title = "Half-Time-Leader's result at Final-Time (leads by 2 or more)"
ML_graph_title = "Effect of Match Location on Winning"
La_Liga_name = "La Liga"
Premier_League_Name = "Premier League"
xticklabels_anal = np.array(["Leader Won", "Draw", "Leader Lost"])
xticklabels_ML = np.array(["Home Team Won", "Draw", "Away Team Won"])
xlabel_anal = "Leader's Status"
xlabel_ML = "Winning teams/Draw"


##### Data Analysis:
def anal_bar_plot_param(league_df):
    print(league_df.columns)
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


def ML_bar_plot_param(league_df):
    league_df_1_Matches = len(league_df)
    H_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'H']) / league_df_1_Matches
    Draws_Percents_1 = len(league_df[league_df['FTR'] == 'D']) / league_df_1_Matches
    A_Wins_Percents_1 = len(league_df[league_df['FTR'] == 'A']) / league_df_1_Matches

    percentages = [H_Wins_Percents_1 * 100,
                           Draws_Percents_1 * 100,
                           A_Wins_Percents_1 * 100]

    return percentages


def two_leagues_bar_plot_comparison(league1_name, league2_name, graph_title, xlabel, xticklabels, x_bar_param_league1, x_bar_param_league2):
    ind = np.arange(3)  # the x locations for the groups
    width = 0.27  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, x_bar_param_league1, width, color='#07a64c')
    rects2 = ax.bar(ind + width * 1.1, x_bar_param_league2, width, color='#b7040e')

    ### Title, Labels, Legend
    ## Title
    plt.title(graph_title)

    ## y label
    ax.set_ylabel("Percents")

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
            ax.text(rect.get_x() + rectWidth / 2., 1.005 * rectHeight, '%g' % round(float(rectHeight), 2), ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ## show
    plt.show()


##### Function Calls:
two_leagues_bar_plot_comparison(La_Liga_name, Premier_League_Name, Analysis_graph1_title, xlabel_anal, xticklabels_anal,
                                anal_bar_plot_param(laLiga0919Filtered2), anal_bar_plot_param(premierLeague9518Filtered2))
two_leagues_bar_plot_comparison(La_Liga_name, Premier_League_Name, Analysis_graph1_title, xlabel_anal, xticklabels_anal,
                                anal_bar_plot_param(laLiga0919Filtered3), anal_bar_plot_param(premierLeague9518Filtered3))
two_leagues_bar_plot_comparison(La_Liga_name, Premier_League_Name, Analysis_graph1_title, xlabel_anal, xticklabels_anal,
                                anal_bar_plot_param(laLiga0919Filtered4), anal_bar_plot_param(premierLeague9518Filtered4))
two_leagues_bar_plot_comparison(La_Liga_name, Premier_League_Name, ML_graph_title, xlabel_ML, xticklabels_ML,
                                ML_bar_plot_param(laLiga0919FilteredML), ML_bar_plot_param(premierLeague9518FilteredML))
