import streamlit as st
import graphviz  # installed with pip
# import classes

# Data manipulation
import pandas as pd
import numpy as np
from simulation import Load, Submission, Tournament
import base64

### change your input dir to a folder with the kaggle data here ###
input_dir = './input_with_data/'

# Info
'''
## NCAA Bracket Builder
Please look at the settings in the side bar to get started. You will add your
solution file there.

This GUI will walk you through the bracket game by game. Periodically a table
will show you your picks so that you can make sure things are on track. If you
want to change your selection for a game, click the radio button for the team
of your choice. The result should be reflected in the text below your choice.
If not try clicking off and on (and be sure to check the table before you
export your results). If you would like a test submission file you can find
one on the [Github page](https://github.com/armstrys/NCAA_BracketBuilder) for
this dashboard.

At the end you will see a table describing your final picks a quick summary of
your bracket that you can screenshot for reference - the bracket is also a
good way to check that your selections are okay. Unfortunately, exporting the
bracket requires a standalone installation of Graphviz and isn't an option at
this time. To make the bracket more visible it is split into the Sweet 16 and
earlier and the Elite 8 and later.

[Streamlit](https://www.streamlit.io/), the tool used to build this allows you
to change the settings to 'wide mode' in the upper right, which makes the
chart more visible. Thanks to Streamlit for building this easy to use tool and
all the folks at Kaggle for putting on the March Machine Learning Mania
[Men's](https://www.kaggle.com/c/ncaam-march-mania-2021) and
[Women's](https://www.kaggle.com/c/ncaaw-march-mania-2021) tournaments!
'''

st.sidebar.write('''
                 If you would like to self-host this app find it on github
                 [here](https://github.com/armstrys/NCAA_BracketBuilder).
                 ''')

# Collect data
sub_file = (st.sidebar
              .file_uploader(label='Drag your Kaggle solution file here'))
mw = (st.sidebar.radio(label='Men\'s or Women\'s submission?',
                       options=['Men', 'Women']))[0][0]

# Prep data
ncaa_files = Load(mw=mw, dir=input_dir)
try:
    sub_df = pd.read_csv(sub_file)
    submission = Submission(sub_df=sub_df, files=ncaa_files)
except ValueError:
    st.warning('''
               Please add a valid solution file and
               adjust settings to continue!
               ''')
    quit()
except KeyError:
    st.warning('''
               Please check that you have selected the
               right competition: men\'s or women\'s'
               ''')
    quit()

# Collect simulation info
season = int(st.sidebar
               .selectbox(label='Season', options=np.sort(submission.seasons)))
season_info = ncaa_files.seasons[ncaa_files.seasons['Season'] == season]
region_dict = {
               'W': season_info['RegionW'].values[0],
               'X': season_info['RegionX'].values[0],
               'Y': season_info['RegionY'].values[0],
               'Z': season_info['RegionZ'].values[0]
                }

style = st.sidebar.radio('Stochastic or Deterministic Bracket?',
                         ['Chalk', 'Random']).lower()
seed = (st.sidebar.number_input(label='Seed for stochastic bracket:',
                                value=0, min_value=0)
        )
np.random.seed(seed)

st.sidebar.write('''
                 **Chalk bracket**: will always select the team favored by the
                 model.\n
                 **Random bracket**: will randomize the winner for each game
                 using the model probabilities. Please choose a new seed to
                 change the randomization!
                 ''')

# Initialize simulation
sim_headers = {
               0: 'First Four',
               1: 'Round of 64',
               2: 'Round of 32',
               3: 'Sweet 16',
               4: 'Elite 8',
               5: 'Final Four',
               6: 'Finals'
              }

tourney = Tournament(ncaa_files, submission, season)

if ncaa_files.mw == 'W':
    tourney.current_r += 1

# Run simulation
while tourney.current_r < 7:

    st.subheader(sim_headers[tourney.current_r])
    tourney.simulate_games(style)
    for g in tourney.games:
        if g.r == tourney.current_r:
            pred = submission.get_pred(g.game_id)
            winner = tourney.results[g.slot]
            if winner.id == g.strong_team.id:
                loser = g.weak_team
            elif winner.id == g.weak_team.id:
                loser = g.strong_team
            else:
                raise ValueError

            w_id = winner.id
            l_id = loser.id
            w_name = winner.name
            l_name = loser.name
            w_prob = pred.proba[w_id]
            l_prob = pred.proba[l_id]
            w_seed = region_dict[winner.seed[0]] + '-' + winner.seed[1:]
            l_seed = region_dict[loser.seed[0]] + '-' + loser.seed[1:]

            st.write(
                f'{w_seed} {w_name} has a {w_prob:.1%}' +
                f' chance of beating {l_seed} {l_name}'
            )
            overwrite = st.radio(label='Manual pick:',
                                 options=[w_name, l_name])
            if overwrite != winner.name:
                w_name = overwrite
                tourney.results.update({g.slot: loser})
            st.write(f'Winner: {w_name}')
    tourney.advance_teams()
    tourney.current_r += 1

results = tourney.results
w_name = results['R6CH'].name
st.write(f'**{w_name} wins the tournament!**')

if mw == 'W':

    odds = tourney.get_odds(submission).values
    bracket_odds = int(1/np.cumprod(odds)[-1])
    avglogloss = np.mean(tourney.get_losses(submission).values)
    success = (odds > .5).sum()/len(odds)

    st.write('''
            According to these probabilities, your odds of a perfect bracket
            based on these selectionsare 1 in **{a:,d}**... Yikes! Good luck!
            :) \n\n The expected logloss of this bracket outcome is {logloss}
            with a model accuracy of {b:,d}%.
            '''.format(a=bracket_odds,
                       logloss=round(avglogloss*1e5)/1e5,
                       b=int(success*100)))
else:

    odds = tourney.get_odds(submission).values[4:]
    bracket_odds = int(1/np.cumprod(odds)[-1])
    avglogloss = np.mean(tourney.get_losses(submission).values[4:])
    success = (odds > .5).sum()/len(odds)

    st.write('''
            According to these probabilities, your odds of a perfect bracket
            based on these selections are 1 in **{a:,d}**... Yikes! Good luck!
            :) \n\n The expected logloss of this bracket outcome is {logloss}
            with a model accuracy of {b:,d}%. These statistics do not include
            play-in games.
            '''.format(a=bracket_odds,
                       logloss=round(avglogloss*1e5)/1e5,
                       b=int(success*100)))


def get_table_download_link():
    '''
    Generates a link allowing the data in a given panda dataframe to be
    downloaded
    in:  dataframe
    out: href string
    '''

    df = pd.DataFrame.from_dict({
                            'slot': results.keys(),
                            'winner': results.values(),
                            'likelihood': tourney.get_odds(submission),
                            'logloss': tourney.get_losses(submission)
                                })

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download results csv</a>'
    return href


st.markdown(get_table_download_link(), unsafe_allow_html=True)


# Graph results
games = tourney.games

early_games = [g for g in games if g.r < 4]
late_games = [g for g in games if g.r >= 4]


def graphGames(games):
    graph = graphviz.Digraph(node_attr={'shape': 'rounded',
                                        'color': 'lightblue2'
                                        })
    for g in games:

        T1 = 'R' + f'{g.r} {g.strong_team.seed}-{g.strong_team.name}'
        T2 = 'R' + f'{g.r} {g.weak_team.seed}-{g.weak_team.name}'
        W = 'R' + f'{g.r+1} {results[g.slot].seed}-{results[g.slot].name}'

        pred = submission.get_pred(f'{season}_' +
                                   f'{g.strong_team.id}_{g.weak_team.id}')
        if g.strong_team.name == results[g.slot].name:
            odds = pred.proba[g.strong_team.id]
            T1_params = {'color': 'green', 'label': f'{odds:.0%}'}
            T2_params = {'color': 'red'}

        else:
            odds = pred.proba[g.weak_team.id]
            T2_params = {'color': 'green', 'label': f'{odds:.0%}'}
            T1_params = {'color': 'red'}

        graph.edge(T1, W, **T1_params)
        graph.edge(T2, W, **T2_params)

    graph.graph_attr['rankdir'] = 'LR'
    graph.graph_attr['size'] = '30'

    graph.node_attr.update(style='rounded')

    return graph


if st.checkbox('Show graphical representation - will slow simulations'):
    st.subheader('Sweet 16 and earlier')
    graph1 = graphGames(early_games)
    st.graphviz_chart(graph1)

    st.subheader('Elite 8 and on')
    graph2 = graphGames(late_games)
    st.graphviz_chart(graph2)
