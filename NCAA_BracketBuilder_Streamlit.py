import streamlit as st
import graphviz as graphviz #installed with pip

## Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path
import glob

path = Path('./input/')  #path for data files

## you can change the other file names here, or just use the Kaggle names and run the dashboard as is.
submission_file = st.sidebar.file_uploader(label='Drag your Kaggle solution file here')
# submission_file = st.sidebar.selectbox(label='Choose submission file from list',options=glob.glob('./input/*submission.csv'))
mw = (st.sidebar.radio(label='Men\'s or Women\'s submission?',options=['Men','Women'] ))[0][0]

seasons_file = path/(mw+'Seasons.csv')
teams_file = path/(mw+'Teams.csv')
seeds_file = path/(mw+'NCAATourneySeeds.csv')
slots_file = path/(mw+'NCAATourneySlots.csv')

#Info
'''
## NCAA Bracket Builder
Please look at the settings in the side bar to get started. You will add your solution file there.

This GUI will walk you through the bracket game by game. Periodically a table will show you your picks (that you can
eventually export) so that you can make sure things are on track. The slider at the top lets you choose a threshold for
games that you want to manually edit. If you only want to edit close games, choose something like 0.6 or 0.7. If you want
all games to be editable choose 1. If you want to change your selection for a game, click the radio button for the team of
your choice. The result should be reflected in the text below your choice. If not try clicking off and on (and be sure to
check the table before you export your results).

At the end you will have the option to export your final picks and you will be able to see a quick summary of your bracket
that you can screenshot for reference - the bracket is also a good way to check that your selections are okay. Unfortunately,
exporting the bracket requires a standalone installation of Graphviz and isn't an option at this time (so have fun squinting).

[Streamlit](https://www.streamlit.io/), the tool used to build this allows you to change the settings to 'wide mode' in the
upper right, which makes the chart more visible. Thanks to Streamlit for building this easy to use tool and all the folks
at Kaggle for putting on the March Machine Learning Mania [Men's]
(https://www.kaggle.com/c/ncaam-march-mania-2021) and [Women's]
(https://www.kaggle.com/c/ncaaw-march-mania-2021) tournaments!
'''

##loading function
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

season_info = pd.read_csv(seasons_file)
teams_dict = pd.read_csv(teams_file).set_index('TeamID')['TeamName'].to_dict() # Create team dictionary to go from team ID to team name
seeds = pd.read_csv(seeds_file)
slots = pd.read_csv(slots_file)

try:
    submission = pd.read_csv(submission_file)
except: 
    st.warning('Please add a valid solution file and adjust settings to continue!')

submission, slots, seeds_dict, season = load_submission(submission,slots,seeds,season_info)

stocastic = st.sidebar.radio('Stocastic or Deterministic Bracket?', ['Deterministic','Stochastic']) == 'Stochastic'
st.sidebar.write('''
                 A deterministic bracket will always select the team favored by the model. The stocastic
                 bracket will randomize the winner for each game using the model probabilities. You will get a
                 different bracket each time you run the model!
                 ''')

games = slots.copy()
games['WinnerSeed'] = ''
games['StrongName'] = ''
games['WeakName'] = ''
games['WinnerName'] = ''
games['StrongID'] = ''
games['WeakID'] = ''
games['WinnerID'] = ''
games.loc[:,'Pred'] = np.nan
game_cols = games.columns.to_list()
new_cols = [game_cols[8]]+game_cols[6:8]+game_cols[4:6]+[game_cols[12]]+game_cols[9:12]+game_cols[0:4]
games = games[new_cols]
games.sort_values('Round',inplace=True)
games.reset_index(inplace=True,drop=True)

def update_games(games,round,next_round):

    for idx,row in games[games['Round']==round].iterrows():
        games.loc[idx,'StrongID'] = seeds_dict[row['StrongSeed']]
        games.loc[idx,'WeakID'] = seeds_dict[row['WeakSeed']]
        games.loc[idx,'StrongName'] = teams_dict[games.loc[idx,'StrongID']]
        games.loc[idx,'WeakName'] = teams_dict[games.loc[idx,'WeakID']]
        games.sort_values(by=['Round','StrongSeed'],inplace=True)

    
    for idx,row in games[games['Round']==round].iterrows():

        if stocastic==True:
            winThresh = np.random.rand()
        else:
            winThresh = .5

        game = row['Game']
        id = (str(season)+'_'+ str(row['StrongID'])+'_'+ str(row['WeakID']))
        pred = submission.loc[submission['ID']==id,'Pred'].values[0]
        if pred> winThresh:
            winslot = row['StrongSeed']
            winID = row['StrongID']
            winname = teams_dict[winID]
            loseslot = row['WeakSeed']
            loseID = row['WeakID']
            losename = teams_dict[loseID]
        else:
            winslot = row['WeakSeed']
            winID = row['WeakID']
            winname = teams_dict[winID]
            loseslot = row['StrongSeed']
            loseID = row['StrongID']
            losename = teams_dict[loseID]
            pred = 1 - pred

        st.subheader( row['StrongSeed'] +' **' + row['StrongName'] + '** vs ' + 
                row['WeakSeed'] + ' **' + row['WeakName'] + '**')
        
        overwrite = st.radio(label='Manual pick:',options=[winname,losename])
        if overwrite == losename:
            winslot = loseslot
            winID = loseID
            winname = losename
            pred = 1 - pred
        

        st.write( str(winname) + ' predicted to win with a ' + str(np.round(pred*1000)/10) + '% chance')
        st.write('**' + winname + '** advances!')
        games.loc[idx,'WinnerSeed'] = winslot
        games.loc[idx,'WinnerID'] = winID
        games.loc[idx,'WinnerName'] = winname
        games.loc[idx,'Pred'] = pred

        if round == 'R0':
            next_slot = game
            games.loc[games['Round']==next_round,'StrongSeed'] = (games.loc[games['Round']==next_round,'StrongSeed']
                                                                    .replace({next_slot:winslot}))
            games.loc[games['Round']==next_round,'WeakSeed'] = (games.loc[games['Round']==next_round,'WeakSeed']
                                                                    .replace({next_slot:winslot}))
        elif round == 'R5':
            if game == 'X':
                games.loc[games['Round']==next_round,'StrongSeed'] = winslot
            else:
                games.loc[games['Round']==next_round,'WeakSeed'] = winslot

        else:
            next_slot = round+game
            games.loc[games['Round']==next_round,'StrongSeed'] = (games.loc[games['Round']==next_round,'StrongSeed']
                                                                    .replace({next_slot:winslot}))
            games.loc[games['Round']==next_round,'WeakSeed'] = (games.loc[games['Round']==next_round,'WeakSeed']
                                                                    .replace({next_slot:winslot}))
    st.write('**Check your picks here before moving on**') 
    st.dataframe(games)

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


if st.button('Export Picks to .csv'):
    games.to_csv(Path('./output/My_NCAA_Bracket.csv'))

st.header('Okay... where\'s my final data? Check your bracket below! Keep scrolling...')
bracket_odds = int(round(1/np.multiply.reduce(np.array(games['Pred']))))
bracket_odds_noPI = int(round(1/np.multiply.reduce(np.array(games.loc[games['Round']!= 'R0','Pred']))))
games['logloss'] = -np.log(games['Pred'])
logloss = np.mean(games.loc[games['Round']!= 'R0','logloss'])

if mw=='W':
    st.write('''
            According to these probabilities, your odds of a perfect bracket are 1 in **{a:,d}**...  
            Yikes! Good luck! :) \n\n The expected logloss of this bracket outcome is {logloss}.
            '''.format(a=bracket_odds,logloss=logloss))
else:
    st.write('''
            According to these probabilities, your odds of a perfect bracket are 1 in **{a:,d}** including
            the play-in games or **{b:,d}** not including the play-in games...  
            Yikes! Good luck! :) \n\n The expected logloss of this bracket outcome is {logloss}.
            '''.format(a=bracket_odds,b=bracket_odds_noPI,logloss=logloss))

## Quick bracket viz
graph = graphviz.Digraph(node_attr={'shape': 'rounded','color': 'lightblue2'})

round_dict = {'R0':'R1',
            'R1':'R2',
            'R2':'R3',
            'R3':'R4',
            'R4':'R5',
            'R5':'R6',
            'R6':'CH',
            'CH':'Winner!'
            }
for _,row in games.iterrows():

    T1 = row['Round']+'-'+row['StrongSeed']+'-'+row['StrongName']
    T2 = row['Round']+'-'+row['WeakSeed']+'-'+row['WeakName']
    W = round_dict[row['Round']]+'-'+row['WinnerSeed']+'-'+row['WinnerName']
    if row['StrongSeed'] == row['WinnerSeed']:

        T1_params = {'color':'green', 'label': (str(int(row['Pred']*100))+'%')}
        T2_params = {'color': 'red'}
        
    else:
        T2_params = {'color':'green', 'label': (str(int(row['Pred']*100))+'%')}
        T1_params = {'color': 'red'}

    graph.edge(T1,W,**T1_params)
    graph.edge(T2,W,**T2_params)

graph.graph_attr['rankdir'] = 'LR'
graph.graph_attr['size'] = '30'

graph.node_attr.update(style='rounded')

st.graphviz_chart(graph)