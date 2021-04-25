import streamlit as st
import graphviz  # installed with pip
# import classes

# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path
import base64

path = Path('./input/')  #path for data files

st.sidebar.write('''
                 If you would like to self-host this app find it on
                 github [here](https://github.com/armstrys/NCAA_BracketBuilder).
                 ''')
submission_file = st.sidebar.file_uploader(label='Drag your Kaggle solution file here')
mw = (st.sidebar.radio(label='Men\'s or Women\'s submission?',options=['Men','Women'] ))[0][0]

seasons_file = path/(mw+'Seasons.csv')
teams_file = path/(mw+'Teams.csv')
seeds_file = path/(mw+'NCAATourneySeeds.csv')
slots_file = path/(mw+'NCAATourneySlots.csv')

#Info
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

## function to load and prep data table for simulations
def load_submission(df,slots,seeds,season_info):
    df[['Season','LeftTeamID','RightTeamID']] = df['ID'].str.split('_',expand=True)
    df.reset_index(inplace=True, drop=True)
    df = df[['ID','Season','LeftTeamID','RightTeamID','Pred']]
    df.columns = ['ID','Season','LeftTeamID','RightTeamID','Pred']
    season = int(st.sidebar.selectbox(label='Season',options=df['Season'].unique()))
    season_info = season_info.loc[season_info['Season']==season].copy()
    region_dict = {'W':season_info['RegionW'].values[0],
                   'X':season_info['RegionX'].values[0],
                   'Y':season_info['RegionY'].values[0],
                   'Z':season_info['RegionZ'].values[0],
                   }

    df_rev = df[['ID','Season','RightTeamID','LeftTeamID','Pred']].copy()
    df_rev.columns = ['ID','Season','LeftTeamID','RightTeamID','Pred']
    df_rev['Pred'] = 1-df_rev['Pred']
    df_rev['ID'] = str(season)+'_'+ df_rev['LeftTeamID'].astype(str)+'_'+df_rev['RightTeamID'].astype(str)
    df = pd.concat([df,df_rev])

    seeds = seeds.loc[seeds['Season']==season,:].copy()
    seeds.drop(columns='Season',inplace=True)
    seeds['Region'] = seeds['Seed'].str.extract(r'([WXYZ]).*')
    seeds['Region'].replace(region_dict,inplace=True)
    seeds['Number'] = seeds['Seed'].str.extract(r'[WXYZ](.*)')
    seeds['NewSeed'] = seeds['Region']+'-'+seeds['Number']
    
    oldseeds_dict = seeds.set_index('Seed')['NewSeed'].to_dict()
    seeds_dict = seeds.set_index('NewSeed')['TeamID'].to_dict()

    if mw == 'W': #womens csv does not have a column for season so we will fake it.
        slots['Season']=season
    else: pass
    slots = slots.loc[slots['Season']==season,:].copy()
    slots.drop(columns='Season',inplace=True)
    slots['StrongSeed'].replace(oldseeds_dict,inplace=True)
    slots['WeakSeed'].replace(oldseeds_dict,inplace=True)
    slots['Round'] = slots['Slot'].str.extract(r'(R.)[WXYZC].').fillna('R0')
    slots['Game'] = slots['Slot'].str.extract(r'.*([WXYZC].*)')

    return df, slots, seeds_dict, season

## load raw data files
season_info = pd.read_csv(seasons_file)
teams_dict = pd.read_csv(teams_file).set_index('TeamID')['TeamName'].to_dict() # Create team dictionary to go from team ID to team name
seeds = pd.read_csv(seeds_file)
slots = pd.read_csv(slots_file)

## try to read submission file
try:
    submission = pd.read_csv(submission_file)
except ValueError:
    st.warning('Please add a valid solution file and adjust settings to continue!')
    quit()

## prep data files for simulation
submission, slots, seeds_dict, season = load_submission(submission,slots,seeds,season_info)

## Choice of bracket modeling type
stochastic = st.sidebar.radio('Stochastic or Deterministic Bracket?', ['Deterministic','Stochastic']) == 'Stochastic'
if stochastic:
    # st.sidebar.button('Generate new stochastic bracket')
    seed = st.sidebar.number_input(label='Seed for stochastic bracket:',value=0,min_value=0)
    np.random.seed(seed)
else: pass

st.sidebar.write('''
                 **Deterministic bracket**: will always select the team favored by the model.\n
                 **Stochastic bracket**: will randomize the winner for each game using the model probabilities.
                 Please choose a new seed to change the randomization!
                 ''')

## preallocate columns for simulation
games = slots.copy()
games['WinnerSeed'] = ''
games['StrongName'] = ''
games['WeakName'] = ''
games['WinnerName'] = ''
games['StrongID'] = ''
games['WeakID'] = ''
games['WinnerID'] = ''
games.loc[:,'WinPred'] = np.nan
games.loc[:,'LogLoss'] = np.nan

game_cols = games.columns.to_list()
new_cols = [game_cols[8]]+game_cols[6:8]+game_cols[4:6]+[game_cols[12]]+game_cols[9:12]+game_cols[0:4]
games = games[new_cols]
games.sort_values('Round',inplace=True)
games.reset_index(inplace=True,drop=True)

## define simulation for one round of tournament games
def update_games(games,rnd,next_rnd):

    ## fill in TeamID's and Names for all games in round
    for idx,row in games[games['Round']==rnd].iterrows():
        games.loc[idx,'StrongID'] = seeds_dict[row['StrongSeed']]
        games.loc[idx,'WeakID'] = seeds_dict[row['WeakSeed']]
        games.loc[idx,'StrongName'] = teams_dict[games.loc[idx,'StrongID']]
        games.loc[idx,'WeakName'] = teams_dict[games.loc[idx,'WeakID']]
        games.sort_values(by=['Round','StrongSeed'],inplace=True)

    ## Model each game
    for idx,row in games[games['Round']==rnd].iterrows():

        ## Set the win threshold to .5 (always take favored team) or random (stochastic model)
        if stochastic==True:
            winThresh = np.random.rand()
        else:
            winThresh = .5

        ## Add a game id
        game = row['Game']
        id = (str(season)+'_'+ str(row['StrongID'])+'_'+ str(row['WeakID']))
        
        ## Get prediction from submission
        try:
            pred = submission.loc[submission['ID']==id,'Pred'].values[0]
        except IndexError:
            st.warning('Please check that you have selected the right competition: men\'s or women\'s')
            quit()

        ## Print matchup to dashboard
        st.subheader( row['StrongSeed'] +' **' + row['StrongName'] + '** vs ' + 
                row['WeakSeed'] + ' **' + row['WeakName'] + '**')
        
        ## determine which team the model favors (may be different than winner in stochastic bracket)
        if pred>.5:
            favoredid = row['StrongID']
            favoredname = teams_dict[favoredid]
        else:
            favoredid = row['WeakID']
            favoredname = teams_dict[favoredid]

        ## set winning and losing team info for game from prediction and win threshold
        if pred > winThresh:
            winslot = row['StrongSeed']
            winID = row['StrongID']
            winname = teams_dict[winID]
            loseslot = row['WeakSeed']
            loseID = row['WeakID']
            losename = teams_dict[loseID]
            winpred = pred
        else:
            winslot = row['WeakSeed']
            winID = row['WeakID']
            winname = teams_dict[winID]
            loseslot = row['StrongSeed']
            loseID = row['StrongID']
            losename = teams_dict[loseID]
            winpred = 1 - pred

        ## option for user to overwrite game - only in deterministic bracket due to streamlit limitations
        # if stochastic==False:
        overwrite = st.radio(label='Manual pick:',options=[winname,losename])
        if overwrite == losename:
            winslot = loseslot
            winID = loseID
            winname = losename
            winpred = 1 - winpred
        else:
            pass # no option to override stochastic matchups since streamlit will re-run everything.
        
        ## Check to see if the current winning team was favored by the model
        if winname==favoredname:
            st.write('The model favors ' + str(favoredname)
                     + ' at ' + str(np.round((.50+abs(pred-.50))*1000)/10) + '%')
        else:
            st.write('The model favors ' + str(favoredname) +
                     ' at ' + str(np.round((.50+abs(pred-.50))*1000)/10) # this accounts for model favoring weak seed
                     + '%, but despite the odds...')

        ## calc logloss
        logloss = -np.log(winpred)

        ## Write out the winner of the game (or tournament!)
        if rnd != 'R6':
            st.write('**' + winname + '** advances! Model log loss = ' + str(round(logloss*1e5)/1e5))
        else:
            st.write('**' + winname + '** wins the ' + str(season) + ' tournament! ' + 
            'Model log loss = '+ str(round(logloss*1e5)/1e5))

        ## placing winner info into data table with their odds of winning
        games.loc[idx,'WinnerSeed'] = winslot
        games.loc[idx,'WinnerID'] = winID
        games.loc[idx,'WinnerName'] = winname
        games.loc[idx,'WinPred'] = winpred
        games.loc[idx,'LogLoss'] = logloss

        ## Placing winner in correct bracket spot depending on round
        if rnd == 'R0': # play in game
            next_slot = game
            games.loc[games['Round']==next_rnd,'StrongSeed'] = (games.loc[games['Round']==next_rnd,'StrongSeed']
                                                                    .replace({next_slot:winslot}))
            games.loc[games['Round']==next_rnd,'WeakSeed'] = (games.loc[games['Round']==next_rnd,'WeakSeed']
                                                                    .replace({next_slot:winslot}))
        elif rnd == 'R5': # Semi-final
            if game == 'X':
                games.loc[games['Round']==next_rnd,'StrongSeed'] = winslot
            else:
                games.loc[games['Round']==next_rnd,'WeakSeed'] = winslot

        else: # all other rounds
            next_slot = rnd+game
            games.loc[games['Round']==next_rnd,'StrongSeed'] = (games.loc[games['Round']==next_rnd,'StrongSeed']
                                                                    .replace({next_slot:winslot}))
            games.loc[games['Round']==next_rnd,'WeakSeed'] = (games.loc[games['Round']==next_rnd,'WeakSeed']
                                                                    .replace({next_slot:winslot}))
    st.write('**Check your picks here before moving on**') 
    st.dataframe(games.dropna())

    return games


if mw == 'M': # no play-in for the womens tourney
    st.header('Play-in games')
    games = update_games(games,'R0','R1')
else: pass

st.header('Let\'s get started!')
st.subheader('Round 1 - Let the madness begin!')
games = update_games(games,'R1','R2')

st.subheader('Round 2 - Are you worn out yet?')
games = update_games(games,'R2','R3')

st.subheader('Round 3 - Sweet 16')
games = update_games(games,'R3','R4')

st.subheader('Round 4 - Elite 8')
games = update_games(games,'R4','R5')

st.subheader('Round 5 - Final 4')
games = update_games(games,'R5','R6')

st.subheader('Round 6 - Championship!')

games = update_games(games,'R6','')

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

st.markdown(get_table_download_link(games), unsafe_allow_html=True)

st.header('Okay... where\'s my final data? Check your bracket below! Keep scrolling...')
bracket_odds = int(round(1/np.multiply.reduce(np.array(games['WinPred']))))
bracket_odds_noPI = int(round(1/np.multiply.reduce(np.array(games.loc[games['Round']!= 'R0','WinPred']))))
avglogloss = np.mean(games.loc[games['Round']!= 'R0','LogLoss'])
success = np.mean(games.loc[games['Round']!= 'R0','WinPred']>.5)

if mw=='W':
    st.write('''
            According to these probabilities, your odds of a perfect bracket are 1 in **{a:,d}**...  
            Yikes! Good luck! :) \n\n The expected logloss of this bracket outcome is {logloss} with a model
            accuracy of {c:,d}%.
            '''.format(a=bracket_odds,
                       logloss=round(avglogloss*1e5)/1e5,
                       c=int(success*100)))
else:
    st.write('''
            According to these probabilities, your odds of a perfect bracket are 1 in **{a:,d}** including
            the play-in games or **{b:,d}** not including the play-in games...  
            Yikes! Good luck! :) \n\n The expected logloss of this bracket outcome is {logloss} with a model
            accuracy of {c:,d}%.
            '''.format(a=bracket_odds,
                       b=bracket_odds_noPI,
                       logloss=round(avglogloss*1e5)/1e5,
                       c=int(success*100)))

## Quick bracket viz

round_dict = {'R0':'R1',
            'R1':'R2',
            'R2':'R3',
            'R3':'R4',
            'R4':'R5',
            'R5':'R6',
            'R6':'CH',
            'CH':'Winner!'
            }

def graphGames(graphrounds):
    graph = graphviz.Digraph(node_attr={'shape': 'rounded','color': 'lightblue2'})

    for _,row in games.loc[games['Round'].isin(graphrounds),:].iterrows():

        T1 = row['Round']+'-'+row['StrongSeed']+'-'+row['StrongName']
        T2 = row['Round']+'-'+row['WeakSeed']+'-'+row['WeakName']
        W = round_dict[row['Round']]+'-'+row['WinnerSeed']+'-'+row['WinnerName']
        if row['StrongSeed'] == row['WinnerSeed']:

            T1_params = {'color':'green', 'label': (str(int(row['WinPred']*100))+'%')}
            T2_params = {'color': 'red'}
            
        else:
            T2_params = {'color':'green', 'label': (str(int(row['WinPred']*100))+'%')}
            T1_params = {'color': 'red'}

        graph.edge(T1,W,**T1_params)
        graph.edge(T2,W,**T2_params)

    graph.graph_attr['rankdir'] = 'LR'
    graph.graph_attr['size'] = '30'

    graph.node_attr.update(style='rounded')

    return graph

st.subheader('Sweet 16 and earlier')
graph1 = graphGames(['R0','R1','R2','R3'] )
st.graphviz_chart(graph1)

st.subheader('Elite 8 and on')
graph2 = graphGames(['R4','R5','R6'] )
st.graphviz_chart(graph2)