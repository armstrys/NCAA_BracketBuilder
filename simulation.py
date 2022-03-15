import re
import numpy as np
from pathlib import Path
import pandas as pd
import itertools
import graphviz  # installed with pip


round_names = {
    0: 'Play-in Games',
    1: 'First Round',
    2: 'Round of 32',
    3: 'Sweet 16',
    4: 'Elite 8',
    5: 'Final 4',
    6: 'Championship'
}


class Data:
    '''
    Loads pertinent data to forward model NCAA tournament
    based on a given probabilities from a submission file.
    Note that the submission file will be handled seperately.
    '''

    def __init__(self, mw=None, dir='./input'):
        if mw is None:
            raise ValueError('Tournament type not set')
        path = Path(dir)
        self.mw = mw.upper()
        self.seasons = pd.read_csv(path/(f'{self.mw}Seasons.csv'))
        self.teams = pd.read_csv(path/(f'{self.mw}Teams.csv'))
        if mw == 'W':
            self.slots = [pd.read_csv(path/(f'WNCAATourneySlots1998thru2021.csv')),
                          pd.read_csv(path/(f'WNCAATourneySlots2022.csv'))]
        else:
            self.slots = pd.read_csv(path/(f'MNCAATourneySlots.csv'))
        self.seeds = pd.read_csv(path/(f'{self.mw}NCAATourneySeeds.csv'))
        self.seedyear_dict, self.seedyear_dict_rev = \
            self.build_seed_dicts()
        self.t_dict = (self.teams.set_index('TeamID')['TeamName']
                                 .to_dict())
        self.t_dict_rev = {v: k for k, v in self.t_dict.items()}

    def build_seed_dicts(self):
        seedyear_dict = {}
        seedyear_dict_rev = {}

        for s in self.seeds['Season'].unique():
            seed_data = self.seeds.query('Season == @s')
            s_dict = (seed_data.set_index('Seed')['TeamID']
                               .to_dict())
            s_dict_rev = {v: k for k, v in s_dict.items()}
            seedyear_dict.update({s: s_dict})
            seedyear_dict_rev.update({s: s_dict_rev})
        return seedyear_dict, seedyear_dict_rev

    def get_round(self, season, t1_id, t2_id):
        return get_round(season,
                         t1_id,
                         t2_id,
                         self.seedyear_dict_rev)


class Team:
    '''
    Simple class to hold team info
    '''

    def __init__(self, t_id, name, seed):
        self.id = t_id
        self.name = name
        self.seed = seed

    def __repr__(self):
        return f'{self.seed} {self.name} - TeamID: {self.id}'


class Submission:
    '''
    Submission is a container for the Prediction class and with
    a method that helps to retrieve predictions based on
    the game_id.
    '''

    def __init__(self, sub_df, data):
        sub_df[['Season', 'Team1ID', 'Team2ID']] = \
            sub_df['ID'].str.split('_', expand=True)
        sub_df[['Season', 'Team1ID', 'Team2ID']] = \
            sub_df[['Season', 'Team1ID', 'Team2ID']].astype(int)
        sub_df['Round'] = \
            sub_df.apply(
                lambda row:
                data.get_round(row['Season'],
                               row['Team1ID'],
                               row['Team2ID']),
                axis=1
            )

        self.seasons = sub_df['Season'].unique().tolist()
        self._df = sub_df.copy()
        self.t_dict = data.t_dict
        self.t_dict_rev = data.t_dict_rev

        def prediction_init(row):
            s_dict = data.seedyear_dict[row['Season']]
            pred = Prediction(row['Season'],
                              row['Team1ID'],
                              row['Team2ID'],
                              row['Pred'],
                              self.t_dict,
                              s_dict)
            return pred

        self._df['PredData'] = self._df.apply(prediction_init, axis=1)

    def get_pred(self, game_id=None):
        '''
        Retrieve prediction using game_id or leave blank to get
        all predictions in a list
        '''
        if game_id is None:
            return self.predictions

        alt_game_id = get_alt_game_id(game_id)
        sub_ids = self.predictions.apply(lambda x: x.game_id)

        idx = ((sub_ids == game_id) |
               (sub_ids == alt_game_id))

        if idx.sum() == 0:
            raise ValueError('Game not found!')

        pred = self.predictions.loc[idx].squeeze()
        return pred

    def get_pred_by_teams(self,
                          season=2021,
                          t1_id=None,
                          t2_id=None,
                          t1_name=None,
                          t2_name=None,):
        ids = False
        if t1_id is not None and t2_id is not None:
            ids = True
        elif t1_name is not None and t2_name is not None:
            if ids:
                raise ValueError(
                    'provide only names or ids of team'
                    )
            t1_id = self.t_dict_rev.get(t1_name)
            t2_id = self.t_dict_rev.get(t2_name)
        else:
            raise ValueError(
                'Please provide a name or ID for both team 1 and 2'
            )
        game_id = f'{season}_{t1_id}_{t2_id}'
        pred = self.lookup_df.loc[game_id, 'PredData']
        return pred

    @property
    def predictions(self):
        return self.df['PredData']

    @property
    def df(self):
        df = self._df.copy()
        df.set_index('ID', inplace=True)
        col_order = ['Season', 'Round', 'Team1ID',
                     'Team2ID', 'Pred', 'PredData']

        return df[col_order]

    @property
    def lookup_df(self):
        df = self.df.copy()
        df_swap = df.copy()
        df_swap.index = df_swap['PredData'].map(
            lambda x: x.alt_game_id
            )
        df_swap.index.name = 'ID'
        df_swap[['Team1ID', 'Team2ID']] = \
            df[['Team2ID', 'Team1ID']].values
        df_swap['Pred'] = 1 - df_swap['Pred']

        return pd.concat([df, df_swap])


class Prediction:
    '''
    Holds the game prediction and methods to query the probability
    or loss for each team. Can also call a winner based on who is
    favored or a random sample using the odds.
    '''

    def __init__(self, season, t1_id, t2_id, pred, t_dict, s_dict):

        self.t_dict = t_dict
        self.s_dict = s_dict
        self.game_id = f'{season}_{t1_id}_{t2_id}'
        self.season = season
        self.t1_id = t1_id
        self.t2_id = t2_id
        self.pred = pred

    def __repr__(self):
        if self.proba[self.t1_id] > .5:
            proba = self.proba[self.t1_id]
            win_name = self.t1_name
            lose_name = self.t2_name
        else:
            proba = self.proba[self.t2_id]
            win_name = self.t2_name
            lose_name = self.t1_name

        return (f'{proba:.1%} chance of '
                f'{win_name} beating {lose_name}')

    @property
    def t1_name(self):
        return self.t_dict[self.t1_id]

    @property
    def t2_name(self):
        return self.t_dict[self.t2_id]

    @property
    def alt_game_id(self):
        return get_alt_game_id(self.game_id)

    @property
    def proba(self):
        return {
            self.t1_id: self.pred,
            self.t2_id: 1 - self.pred
        }

    @property
    def logloss(self):
        return {
            self.t1_id: -np.log(self.pred),
            self.t2_id: -np.log(1 - self.pred)
        }

    @property
    def round(self):
        return self.get_round

    def get_favored(self):
        if self.proba[self.t1_id] > 0.5:
            return self.t1_id
        else:
            return self.t2_id

    def get_random(self):
        if self.proba[self.t1_id] > np.random.rand():
            return self.t1_id
        else:
            return self.t2_id


class Tournament:

    def __init__(self, data, submission, season):
        '''
        Tournament class will hold all game classes and functions
        to handle games. Needs NCAA data and submission as input
        along with the season that will be modeled.
        '''

        # Add metadata to be called by class
        self.submission = submission  # submission class to get preds
        self.mw = data.mw
        self.season = season  # season year
        self.current_r = 0  # initiate at round 0 (play-in)
        self.results = {}  # results stored as slot: TeamID

        # Create seed: teamID dictionary
        self.t_dict = data.t_dict
        self.s_dict = data.seedyear_dict[self.season]
        self.s_dict_rev = data.seedyear_dict_rev[self.season]
        self._summary = {}

        # Only men's file has differing slots by year - select the year
        #       we need and remove season column
        slots = data.slots
        if self.mw == 'M':
            slots = slots[data.slots['Season'] == season].copy()
            slots.drop(columns='Season', inplace=True)
        else:
            if season < 2022:
                slots = slots[0]
            else:
                slots = slots[1]

        # Initiate game classes and save as Tournament attribute
        def game_init(row):
            game = Game(row, submission, data.t_dict,
                        self.s_dict, self.season)
            return game

        if (len(self.s_dict) == 0) or (len(slots) == 0):
            raise RuntimeError('''
                    Please check to see that your submission file and
                    tournament data class has both have the appropriate season.
                    ''')
        self.games = slots.apply(game_init, axis=1)
        self.games.index = slots['Slot']

    @property
    def n_teams(self):
        return len(self.s_dict)

    @property
    def summary(self):
        if len(self._summary) == 0:
            self._summary = self.summarize_results()
        return self._summary

    def summarize_results(self, previous_summary=None):
        if previous_summary is not None:
            self._summary = previous_summary
        for slot, team in self.results.items():
            team = team.id
            r = slot[:2]
            if 'R' not in r:
                r = 'R0'
            if self._summary.get(r) is None:
                self._summary.update({r: {team: 1}})
            elif self._summary[r].get(team) is None:
                self._summary[r].update({team: 1})
            else:
                self._summary[r][team] += 1
        return self._summary

    def summary_to_df(self, summary=None, n_sim=1):

        if summary is None:
            summary = self.summary

        columns = [round_names.get(k) for k in range(7)]
        if self.mw == 'W' and self.season < 2022:
            columns = columns[1:]
        summary_df = pd.DataFrame(summary)
        summary_df.columns = columns
        summary_df.index.name = 'TeamID'
        all_teams = list(self.s_dict.values())
        missing_teams = list(set(all_teams) - set(summary_df.index))
        if len(missing_teams) > 0:
            missing_teams_df = pd.DataFrame(np.nan,
                                            index=missing_teams,
                                            columns=summary_df.columns)
            summary_df = pd.concat([summary_df, missing_teams_df])
        summary_df['Team'] = [f'{self.s_dict_rev[t]} - '
                              f'{self.submission.t_dict[t]}'
                              for t in summary_df.index]
        columns.insert(0, 'Team')
        summary_df = summary_df[columns]

        summary_df['First Round'].fillna(n_sim, inplace=True)
        summary_df.fillna(0, inplace=True)
        summary_df.sort_values(by=columns[::-1], ascending=False, inplace=True)
        return summary_df

    def simulate_games(self, style):
        '''
        This function uses each game class from a specific round
        to predict the winner (set to either the favorite or
        a random sample based on the odds). The winner is added to
        the results file under the appropriate tournament slot.
        To advance teams to the next round us advance_teams()
        afer this or run simulate_round(), which runs both.
        '''

        # function pull predicted result from game if in same round
        def find_winner(x):
            if x.r == self.current_r:
                win_id = x.get_winner(self.submission, style)
                if x.strong_team.id == win_id:
                    self.results.update({x.slot: x.strong_team})
                elif x.weak_team.id == win_id:
                    self.results.update({x.slot: x.weak_team})
                else:
                    raise ValueError('Couldn\'t find winner')

            else:
                pass

        self.games.apply(find_winner)  # apply function

    def advance_teams(self):
        '''
        calls on all tournament games to update their slots
        based on results dictionary
        '''

        self.games.apply(lambda x: x.add_teams(self.results))

    def simulate_round(self, style):
        '''
        runs the appropriate functions to both simulate the
        current round and advances teams. Note that round needs
        to be manually incremented and can be found in
        Tournament.current_r. If you want to sim the whole tourney
        and you don't need to alter data in between rounds just
        use simulate_tournament()
        '''

        self.simulate_games(style)
        self.advance_teams()

    def simulate_tournament(self, style, seed=None):
        '''
        Runs single round simulation until all are complete.
        '''
        if seed is not None:
            np.random.seed(seed)  # seed np at tournament level

        # Run simulations for round 0->6
        while self.current_r < 7:
            self.simulate_round(style)
            self.current_r += 1  # increments round by 1

    def simulate_tournaments(self, n_sim=500):
        '''
        Puts the tournament results in a summary format keyed
        by team and round that can be aggregated over multiple
        simulations. This results dict can be made into a pandas
        dataframe by simply calling pd.DataFrame(results) on
        a results dictionary that holds simulated outputs.
        args:
            summary_dict: (dict) if running multiple simultaions
                put the previous summary result as an argument
                to iteratively
        '''

        summary = {}
        expected_losses = []

        for i in range(int(n_sim)):
            self.reset_tournament()
            self.simulate_tournament('random', seed=i)
            summary = self.summarize_results(previous_summary=summary)
            losses = self.get_losses(kaggle=True)
            loss = losses.mean()
            expected_losses.append(loss)

        self._summary = summary
        self.expected_losses = np.array(expected_losses)
        return self.summary_to_df(self._summary, n_sim=n_sim), \
            self.expected_losses

    def get_losses(self, kaggle=True):
        '''
        gets losses for all predictions based on the results
        dictionary

        Kaggle=True to exlude play-ins
        '''

        def logloss(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = self.submission.get_pred(game_id)
            logloss = pred.logloss[w_id]
            return logloss

        losses = self.games.apply(lambda x: logloss(x))
        if kaggle:
            losses = losses.loc[
                    losses.index.str.startswith('R')
                    ]

        return losses

    def get_odds(self, kaggle=True):
        '''
        gets odds for all predictions based on the results
        dictionary
        '''

        def calc_odds(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = self.submission.get_pred(game_id)
            proba = pred.proba[w_id]
            return proba

        odds = self.games.apply(lambda x: calc_odds(x))
        if kaggle:
            odds = odds.loc[
                    odds.index.str.startswith('R')
                    ]
        return odds

    def graph_games(self, rounds=list(range(7))):
        games = [g for g in self.games if g.r in rounds]

        graph = graphviz.Digraph(node_attr={'shape': 'rounded',
                                            'color': 'lightblue2'
                                            })
        for g in games:

            T1 = 'R' + f'{g.r} {g.strong_team.seed}-{g.strong_team.name}'
            T2 = 'R' + f'{g.r} {g.weak_team.seed}-{g.weak_team.name}'
            W = 'R' + f'{g.r+1} {self.results[g.slot].seed}' \
                f'-{self.results[g.slot].name}'

            pred = self.submission.get_pred(f'{self.season}_'
                                            f'{g.strong_team.id}_'
                                            f'{g.weak_team.id}')
            if g.strong_team.name == self.results[g.slot].name:
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

    def update_results(self, new_results):
        '''
        method to update the results dict with a generic
        slots: team_id dict
        '''
        self.reset_tournament()
        new_results_team = \
            {slot: Team(t_id=tid,
                        name=self.t_dict.get(tid),
                        seed=self.s_dict_rev.get(tid)
                        )
             for slot, tid in new_results.items()}
        self.results.update(new_results_team)

        self.advance_teams()
        self._summary = {}

    def get_historic_results(self):
        self.update_results(historic_results[self.mw][self.season])

    def reset_tournament(self):
        self.current_r = 0  # initiate at round 0 (play-in)
        self.results = {}  # results stored as slot: TeamID
        self._summary = {}


class Game:
    '''
    Game class is an object for each tournament slot that is
    populated as the tournament continues. It also holds functions
    relavent to a game like updating teams from the results dict
    and returning a winner based on the predictions from the
    submission class.
    '''

    def __init__(self, row_slots, submission, t_dict, s_dict, season):
        # Add relavent metadata for game - source is slots csv
        self.season = season
        self.slot = row_slots['Slot']
        self.strong_seed = row_slots['StrongSeed']
        self.weak_seed = row_slots['WeakSeed']

        # extract round label from game
        r = re.compile(r'(R.)[WXYZC].')
        match = r.search(self.slot)
        if match is not None:
            self.r_label = match.group(1)
        else:
            self.r_label = 'R0'  # label play-in games

        # set round equiv to tournament.current_r (int)
        self.r = int(self.r_label[-1])

        # Set teams if slot is determined only by seed
        #       This places only the initial games.
        self.strong_team = None
        self.weak_team = None
        strong_id = s_dict.get(self.strong_seed)
        weak_id = s_dict.get(self.weak_seed)

        # Initiate team class that holds team attrib.
        if strong_id is not None:
            self.strong_team = Team(strong_id,
                                    t_dict.get(strong_id),
                                    self.strong_seed)

        if weak_id is not None:
            self.weak_team = Team(weak_id,
                                  t_dict.get(weak_id),
                                  self.weak_seed)

    def __repr__(self):
        if self.team_is_missing():
            return f'{self.season} - {self.slot}: Game not yet set'
        else:
            return (f'{self.season} - {self.slot}: {self.strong_team.name} '
                    f'vs. {self.weak_team.name}')

    @property
    def game_id(self):
        return '_'.join([str(self.season),
                         str(self.strong_team.id),
                         str(self.weak_team.id)])

    def add_teams(self, results):
        '''
        Checks all results and updates games if results exist.
        '''

        if results.get(self.strong_seed) is not None:
            self.strong_team = results.get(self.strong_seed)
        if results.get(self.weak_seed) is not None:
            self.weak_team = results.get(self.weak_seed)

    def team_is_missing(self):
        '''
        Checks if either team is missing
        '''
        if self.strong_team is None or self.weak_team is None:
            return True
        else:
            return False

    def get_winner(self, submission, style, seed=0):
        '''
        Retrieves the winner of the game from the submission
        file based on the chosen methodology.
        '''

        if self.team_is_missing():
            raise ValueError('At least one team does not exist')

        if style == 'chalk':
            win_id = (
                submission.get_pred(self.game_id)
                          .get_favored()
                          )
            return win_id
        elif style == 'random':
            win_id = (
                submission.get_pred(self.game_id)
                          .get_random()
                          )
            return win_id
        else:
            raise ValueError('Please choose style=random or chalk')


def get_alt_game_id(game_id):
    alt_game_id = game_id.split('_')
    alt_game_id = '_'.join([alt_game_id[0],
                            alt_game_id[2],
                            alt_game_id[1]
                            ])
    return alt_game_id


def get_round(season, t1_id, t2_id, seedyear_dict_rev):
    round_dict = gen_round_dict()

    s_dict_rev = seedyear_dict_rev[season]
    t1_seed = s_dict_rev[t1_id]
    t2_seed = s_dict_rev[t2_id]

    t1_seednum = int(t1_seed[1:3])
    t2_seednum = int(t2_seed[1:3])

    t1_reg = t1_seed[0]
    t2_reg = t2_seed[0]

    area_dict = {'W':'WX', 'X':'WX', 'Y':'YZ', 'Z':'YZ'}

    t1_area = area_dict.get(t1_reg)
    t2_area = area_dict.get(t2_reg)

    if t1_area != t2_area:
        return 6
    elif t1_reg != t2_reg:
        return 5
    else:
        matchup = f'{t2_seednum}v{t1_seednum}'
        return round_dict.get(matchup)
    raise 

def gen_round_dict():
    round_dict = {}

    r4 = [[1,16,8,9,5,12,4,13,6,11,3,14,7,10,15,2]]
    for seeds in r4:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 4
            round_dict[str(pair[1])+'v'+str(pair[0])] = 4


    r3 = [[1,16,8,9,5,12,4,13],[6,11,3,14,7,10,15,2]]
    for seeds in r3:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 3
            round_dict[str(pair[1])+'v'+str(pair[0])] = 3

    r2 = [[1,16,8,9],[5,12,4,13],[6,11,3,14],[7,10,15,2]]
    for seeds in r2:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 2
            round_dict[str(pair[1])+'v'+str(pair[0])] = 2

    r1 = [[1,16],[8,9],[5,12],[4,13],[6,11],[3,14],[7,10],[15,2]]
    for seeds in r1:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 1
            round_dict[str(pair[1])+'v'+str(pair[0])] = 1

    round_dict['11v11'] = 0
    round_dict['12v12'] = 0
    round_dict['13v13'] = 0
    round_dict['14v14'] = 0
    round_dict['16v16'] = 0
    return round_dict

historic_results = {
    'M': {
            2016: {
            "W11":1276,
            "W16":1195,
            "Y11":1455,
            "Z16":1221,
            "R1W1":1314,
            "R1W2":1462,
            "R1W3":1372,
            "R1W4":1246,
            "R1W5":1231,
            "R1W6":1323,
            "R1W7":1458,
            "R1W8":1344,
            "R1X1":1438,
            "R1X2":1292,
            "R1X3":1428,
            "R1X4":1235,
            "R1X5":1114,
            "R1X6":1211,
            "R1X7":1393,
            "R1X8":1139,
            "R1Y1":1242,
            "R1Y2":1437,
            "R1Y3":1274,
            "R1Y4":1218,
            "R1Y5":1268,
            "R1Y6":1455,
            "R1Y7":1234,
            "R1Y8":1163,
            "R1Z1":1332,
            "R1Z2":1328,
            "R1Z3":1401,
            "R1Z4":1181,
            "R1Z5":1463,
            "R1Z6":1320,
            "R1Z7":1433,
            "R1Z8":1386,
            "R2W1":1314,
            "R2W2":1458,
            "R2W3":1323,
            "R2W4":1231,
            "R2X1":1438,
            "R2X2":1393,
            "R2X3":1211,
            "R2X4":1235,
            "R2Y1":1242,
            "R2Y2":1437,
            "R2Y3":1274,
            "R2Y4":1268,
            "R2Z1":1332,
            "R2Z2":1328,
            "R2Z3":1401,
            "R2Z4":1181,
            "R3W1":1314,
            "R3W2":1323,
            "R3X1":1438,
            "R3X2":1393,
            "R3Y1":1242,
            "R3Y2":1437,
            "R3Z1":1332,
            "R3Z2":1328,
            "R4W1":1314,
            "R4X1":1393,
            "R4Y1":1437,
            "R4Z1":1328,
            "R5WX":1314,
            "R5YZ":1437,
            "R6CH":1437,
        },
        2017: {
            "W11":1425,
            "W16":1291,
            "Y16":1413,
            "Z11":1243,
            "R1W1":1437,
            "R1W2":1181,
            "R1W3":1124,
            "R1W4":1196,
            "R1W5":1438,
            "R1W6":1425,
            "R1W7":1376,
            "R1W8":1458,
            "R1X1":1211,
            "R1X2":1112,
            "R1X3":1199,
            "R1X4":1452,
            "R1X5":1323,
            "R1X6":1462,
            "R1X7":1388,
            "R1X8":1321,
            "R1Y1":1242,
            "R1Y2":1257,
            "R1Y3":1332,
            "R1Y4":1345,
            "R1Y5":1235,
            "R1Y6":1348,
            "R1Y7":1276,
            "R1Y8":1277,
            "R1Z1":1314,
            "R1Z2":1246,
            "R1Z3":1417,
            "R1Z4":1139,
            "R1Z5":1292,
            "R1Z6":1153,
            "R1Z7":1455,
            "R1Z8":1116,
            "R2W1":1458,
            "R2W2":1376,
            "R2W3":1124,
            "R2W4":1196,
            "R2X1":1211,
            "R2X2":1112,
            "R2X3":1462,
            "R2X4":1452,
            "R2Y1":1242,
            "R2Y2":1276,
            "R2Y3":1332,
            "R2Y4":1345,
            "R2Z1":1314,
            "R2Z2":1246,
            "R2Z3":1417,
            "R2Z4":1139,
            "R3W1":1196,
            "R3W2":1376,
            "R3X1":1211,
            "R3X2":1462,
            "R3Y1":1242,
            "R3Y2":1332,
            "R3Z1":1314,
            "R3Z2":1246,
            "R4W1":1376,
            "R4X1":1211,
            "R4Y1":1332,
            "R4Z1":1314,
            "R5WX":1211,
            "R5YZ":1314,
            "R6CH":1314
        },
        2018: {
            "W11":1382,
            "W16":1347,
            "X11":1393,
            "Z16":1411,
            "R1W1":1437,
            "R1W2":1345,
            "R1W3":1403,
            "R1W4":1267,
            "R1W5":1452,
            "R1W6":1196,
            "R1W7":1139,
            "R1W8":1104,
            "R1X1":1242,
            "R1X2":1181,
            "R1X3":1277,
            "R1X4":1120,
            "R1X5":1155,
            "R1X6":1393,
            "R1X7":1348,
            "R1X8":1371,
            "R1Y1":1420,
            "R1Y2":1153,
            "R1Y3":1397,
            "R1Y4":1138,
            "R1Y5":1246,
            "R1Y6":1260,
            "R1Y7":1305,
            "R1Y8":1243,
            "R1Z1":1462,
            "R1Z2":1314,
            "R1Z3":1276,
            "R1Z4":1211,
            "R1Z5":1326,
            "R1Z6":1222,
            "R1Z7":1401,
            "R1Z8":1199,
            "R2W1":1437,
            "R2W2":1345,
            "R2W3":1403,
            "R2W4":1452,
            "R2X1":1242,
            "R2X2":1181,
            "R2X3":1393,
            "R2X4":1155,
            "R2Y1":1243,
            "R2Y2":1305,
            "R2Y3":1260,
            "R2Y4":1246,
            "R2Z1":1199,
            "R2Z2":1401,
            "R2Z3":1276,
            "R2Z4":1211,
            "R3W1":1437,
            "R3W2":1403,
            "R3X1":1242,
            "R3X2":1181,
            "R3Y1":1243,
            "R3Y2":1260,
            "R3Z1":1199,
            "R3Z2":1276,
            "R4W1":1437,
            "R4X1":1242,
            "R4Y1":1260,
            "R4Z1":1276,
            "R5WX":1437,
            "R5YZ":1276,
            "R6CH":1437
        },
        2019: {
            "W11":1125,
            "W16":1295,
            "X11":1113,
            "X16":1192,
            "R1W1":1181,
            "R1W2":1277,
            "R1W3":1261,
            "R1W4":1439,
            "R1W5":1251,
            "R1W6":1268,
            "R1W7":1278,
            "R1W8":1416,
            "R1X1":1211,
            "R1X2":1276,
            "R1X3":1403,
            "R1X4":1199,
            "R1X5":1293,
            "R1X6":1138,
            "R1X7":1196,
            "R1X8":1124,
            "R1Y1":1314,
            "R1Y2":1246,
            "R1Y3":1222,
            "R1Y4":1242,
            "R1Y5":1120,
            "R1Y6":1326,
            "R1Y7":1459,
            "R1Y8":1449,
            "R1Z1":1438,
            "R1Z2":1397,
            "R1Z3":1345,
            "R1Z4":1414,
            "R1Z5":1332,
            "R1Z6":1437,
            "R1Z7":1234,
            "R1Z8":1328,
            "R2W1":1181,
            "R2W2":1277,
            "R2W3":1261,
            "R2W4":1439,
            "R2X1":1211,
            "R2X2":1276,
            "R2X3":1403,
            "R2X4":1199,
            "R2Y1":1314,
            "R2Y2":1246,
            "R2Y3":1222,
            "R2Y4":1120,
            "R2Z1":1438,
            "R2Z2":1397,
            "R2Z3":1345,
            "R2Z4":1332,
            "R3W1":1181,
            "R3W2":1277,
            "R3X1":1211,
            "R3X2":1403,
            "R3Y1":1120,
            "R3Y2":1246,
            "R3Z1":1438,
            "R3Z2":1345,
            "R4W1":1277,
            "R4X1":1403,
            "R4Y1":1120,
            "R4Z1":1438,
            "R5WX":1403,
            "R5YZ":1438,
            "R6CH":1438
        },
        2021: {
            "W11":1417, # this is a play-in game
            "W16":1411, # this is a play-in game
            "X11":1179, # this is a play-in game
            "X16":1313, # this is a play-in game
            "R1W1":1276,
            "R1W2":1104,
            "R1W3":1101,
            "R1W4":1199,
            "R1W5":1160,
            "R1W6":1417,
            "R1W7":1268,
            "R1W8":1261,
            "R1X1":1211,
            "R1X2":1234,
            "R1X3":1242,
            "R1X4":1325, # this game didnt happen!
            "R1X5":1166,
            "R1X6":1425,
            "R1X7":1332,
            "R1X8":1328,
            "R1Y1":1228,
            "R1Y2":1222,
            "R1Y3":1452,
            "R1Y4":1329,
            "R1Y5":1333,
            "R1Y6":1393,
            "R1Y7":1353,
            "R1Y8":1260,
            "R1Z1":1124,
            "R1Z2":1331,
            "R1Z3":1116,
            "R1Z4":1317,
            "R1Z5":1437,
            "R1Z6":1403,
            "R1Z7":1196,
            "R1Z8":1458,
            "R2W1":1276,
            "R2W2":1104,
            "R2W3":1417,
            "R2W4":1199,
            "R2X1":1211,
            "R2X2":1332,
            "R2X3":1425,
            "R2X4":1166,
            "R2Y1":1260,
            "R2Y2":1222,
            "R2Y3":1393,
            "R2Y4":1333,
            "R2Z1":1124,
            "R2Z2":1331,
            "R2Z3":1116,
            "R2Z4":1437,
            "R3W1":1276,
            "R3W2":1417,
            "R3X1":1211,
            "R3X2":1425,
            "R3Y1":1333,
            "R3Y2":1222,
            "R3Z1":1124,
            "R3Z2":1116,
            "R4W1":1417,
            "R4X1":1211,
            "R4Y1":1222,
            "R4Z1":1124,
            "R5WX":1211,
            "R5YZ":1124,
            "R6CH":1124,
        }
    },
    'W': {}
}
