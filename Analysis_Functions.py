import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms

from Leagues_Data_and_Adaptations import laLiga0919Filtered2, laLiga0919Filtered3, laLiga0919Filtered4, laLiga0919FilteredML
from Leagues_Data_and_Adaptations import premierLeague9518Filtered2, premierLeague9518Filtered3, premierLeague9518Filtered4, premierLeague9518FilteredML

plt.style.use(['seaborn-white', 'bmh'])

Analysis_graph1_title = "Half-Time-Leader's result at Final-Time"
Analysis_graph2_title = "Half-Time-Leader's result at Final-Time (leads by exactly 1)"
Analysis_graph3_title = "Half-Time-Leader's result at Final-Time (leads by 2 or more)"

##### Analysis Stage:
#### Bar graph of Half-Time-leader's status at Final Time result
def HT_leader_status_at_FT_bar_plot_comparison(league_df_1, league_df_2, graph_title):
    ind = np.arange(3)  # the x locations for the groups
    width = 0.27  # the width of the bars
    leaderStatus = np.array(["Leader Won", "Draw", "Leader Lost"])

    ### parameters for 1st league:
    num_of_games_that_have_lead_at_HT_1 = len(league_df_1)
    num_of_games_that_leader_wins_at_FT_1 = len(league_df_1[league_df_1.FTR == league_df_1.HTR])
    num_of_games_that_draw_at_FT_1 = len(league_df_1[league_df_1.FTR == 'D'])
    num_of_games_that_leader_loses_at_FT_1 = num_of_games_that_have_lead_at_HT_1 - \
                                             num_of_games_that_leader_wins_at_FT_1 - \
                                             num_of_games_that_draw_at_FT_1
    percentages_for_1st = [(num_of_games_that_leader_wins_at_FT_1 / num_of_games_that_have_lead_at_HT_1) * 100,
                           (num_of_games_that_draw_at_FT_1 / num_of_games_that_have_lead_at_HT_1) * 100,
                           (num_of_games_that_leader_loses_at_FT_1 / num_of_games_that_have_lead_at_HT_1) * 100]

    ### parameters for 2nd league:
    num_of_games_that_have_lead_at_HT_2 = len(league_df_2)
    num_of_games_that_leader_wins_at_FT_2 = len(league_df_2[league_df_2.FTR == league_df_2.HTR])
    num_of_games_that_draw_at_FT_2 = len(league_df_2[league_df_2.FTR == 'D'])
    num_of_games_that_leader_loses_at_FT_2 = num_of_games_that_have_lead_at_HT_2 - \
                                             num_of_games_that_leader_wins_at_FT_2 - \
                                             num_of_games_that_draw_at_FT_2
    percentages_for_2nd = [(num_of_games_that_leader_wins_at_FT_2 / num_of_games_that_have_lead_at_HT_2) * 100,
                           (num_of_games_that_draw_at_FT_2 / num_of_games_that_have_lead_at_HT_2) * 100,
                           (num_of_games_that_leader_loses_at_FT_2 / num_of_games_that_have_lead_at_HT_2) * 100]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, percentages_for_1st, width, color='#07a64c')
    rects2 = ax.bar(ind + width * 1.1, percentages_for_2nd, width, color='#b7040e')

    ### Title, Labels, Legend
    ## Title
    plt.title(graph_title)

    ## y label
    ax.set_ylabel("Percents")

    ## x label
    plt.xlabel("Leader's Status")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(leaderStatus)

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
    ax.legend((rects1[0], rects2[0]), ("La Liga", "Premier League"))
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


##### ML Stage:
#### Effect of match location on winning bar graph
ML_by_location_graph_title = 'Effect of Match Location on Winning'

def winning_chances_by_location_bar_plot_comparison(league_df_1, league_df_2, graph_title):
    ind = np.arange(3)  # the x locations for the groups
    width = 0.27  # the width of the bars
    leaderStatus = np.array(["Home Team Won", "Draw", "Away Team Won"])

    ### parameters for 1st league:
    league_df_1_Matches = len(league_df_1)
    H_Wins_Percents_1 = len(league_df_1[league_df_1['FTR'] == 'H']) / league_df_1_Matches
    Draws_Percents_1 = len(league_df_1[league_df_1['FTR'] == 'D']) / league_df_1_Matches
    A_Wins_Percents_1 = len(league_df_1[league_df_1['FTR'] == 'A']) / league_df_1_Matches

    percentages_for_1st = [H_Wins_Percents_1 * 100,
                           Draws_Percents_1 * 100,
                           A_Wins_Percents_1 * 100]

    ### parameters for 2nd league:
    league_df_2_Matches = len(league_df_2)
    H_Wins_Percents_2 = len(league_df_2[league_df_2['FTR'] == 'H']) / league_df_2_Matches
    Draws_Percents_2 = len(league_df_2[league_df_2['FTR'] == 'D']) / league_df_2_Matches
    A_Wins_Percents_2 = len(league_df_2[league_df_2['FTR'] == 'A']) / league_df_2_Matches

    percentages_for_2nd = [H_Wins_Percents_2 * 100,
                           Draws_Percents_2 * 100,
                           A_Wins_Percents_2 * 100]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, percentages_for_1st, width, color='#07a64c')
    rects2 = ax.bar(ind + width * 1.1, percentages_for_2nd, width, color='#b7040e')

    ### Title, Labels, Legend
    ## Title
    plt.title(graph_title)

    ## y label
    ax.set_ylabel("Percents")

    ## x label
    plt.xlabel("Winning teams/Draw")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(leaderStatus)

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
    ax.legend((rects1[0], rects2[0]), ("La Liga", "Premier League"))
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
# winning_chances_by_location_bar_plot_comparison(laLiga0919FilteredML, premierLeague9518FilteredML, ML_by_location_graph_title)
HT_leader_status_at_FT_bar_plot_comparison(laLiga0919Filtered2, premierLeague9518Filtered2, Analysis_graph1_title)
# HT_leader_status_at_FT_bar_plot_comparison(laLiga0919Filtered3, premierLeague9518Filtered3, Analysis_graph2_title)
# HT_leader_status_at_FT_bar_plot_comparison(laLiga0919Filtered4, premierLeague9518Filtered4, Analysis_graph3_title)
