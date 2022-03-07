import re
import numpy as np
from pathlib import Path
import pandas as pd


class Load:
    '''
    Loads pertinent files to forward model NCAA tournament
    based on a given probabilities from a submission file.
    Note that the submission file will be handled seperately.
    '''

    def __init__(self, mw=None, dir='./input'):
        self.mw = mw
        if mw is None:
            raise ValueError('Tournament type not set')
        path = Path(dir)
        self.seasons = pd.read_csv(path/(mw+'Seasons.csv'))
        self.teams = pd.read_csv(path/(mw+'Teams.csv'))
        self.slots = pd.read_csv(path/(mw+'NCAATourneySlots.csv'))
        self.seeds = pd.read_csv(path/(mw+'NCAATourneySeeds.csv'))
        self.t_dict = (pd.read_csv(path/(mw+'Teams.csv'))
                         .set_index('TeamID')['TeamName']
                         .to_dict())
        self.t_dict_rev = (pd.read_csv(path/(mw+'Teams.csv'))
                             .set_index('TeamName')['TeamID']
                             .to_dict())


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

    def __init__(self, sub_df, files):

        df = sub_df.copy()
        df.columns = [s.lower() for s in df.columns]

        self.seasons = df['id'].apply(lambda x: int(x[0:4])).unique()
        self.t_dict = files.t_dict

        def prediction_init(row):
            pred = Prediction(row, self.t_dict)
            return pred

        self.predictions = df.apply(prediction_init, axis=1)

    def get_pred(self, game_id=None):
        '''
        Retrieve prediction using game_id or leave blank to get
        all predictions in a list
        '''

        alt_game_id = game_id.split('_')
        alt_game_id = '_'.join([alt_game_id[0],
                                alt_game_id[2],
                                alt_game_id[1]
                                ])

        sub_ids = self.predictions.apply(lambda x: x.game_id)

        idx = ((sub_ids == game_id) |
               (sub_ids == alt_game_id))

        if idx.sum() == 0:
            raise ValueError('Game not found!')

        pred = self.predictions.loc[idx].squeeze()
        return pred


class Prediction:
    '''
    Holds the game prediction and methods to query the probability
    or loss for each team. Can also call a winner based on who is
    favored or a random sample using the odds.
    '''

    def __init__(self, sub_row, t_dict):

        self.game_id = sub_row['id']
        self.season, self.t1_id, self.t2_id = (
            [int(x) for x in self.game_id.split('_')]
        )

        self.t1_name = t_dict[self.t1_id]
        self.t2_name = t_dict[self.t2_id]

        self.proba = {
                      self.t1_id: sub_row['pred'],
                      self.t2_id: 1 - sub_row['pred']
                     }

        self.logloss = {
                        self.t1_id: -np.log(sub_row['pred']),
                        self.t2_id: -np.log(1 - sub_row['pred'])
                       }

    def __repr__(self):
        if self.proba[self.t1_id] > .5:
            proba = self.proba[self.t1_id]
            win_name = self.t1_name
            lose_name = self.t2_name
        else:
            proba = self.proba[self.t2_id]
            win_name = self.t2_name
            lose_name = self.t1_name

        return (f'{proba:.1%} chance of ' +
                f'{win_name} beating {lose_name}')

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

    def __init__(self, files, submission, season):
        '''
        Tournament class will hold all game classes and functions
        to handle games. Needs NCAA files and submission as input
        along with the season that will be modeled.
        '''

        # Add metadata to be called by class
        self.submission = submission  # submission class to get preds
        self.season = season  # season year
        self.current_r = 0  # initiate at round 0 (play-in)
        self.results = {}  # results stored as slot: TeamID

        # Create seed: teamID dictionary
        seeds = files.seeds[files.seeds['Season'] == self.season]
        self.s_dict = (seeds.set_index('Seed')['TeamID']
                            .to_dict())
        self.s_dict_rev = (seeds.set_index('TeamID')['Seed']
                                .to_dict())

        # Only men's file has differing slots by year - select the year
        #       we need and remove season column
        slots = files.slots
        if 'Season' in slots.columns:
            slots = slots[files.slots['Season'] == season].copy()
            slots.drop(columns='Season', inplace=True)
        else:
            pass

        # Initiate game classes and save as Tournament attribute
        def game_init(row):
            game = Game(row, submission, files.t_dict,
                        self.s_dict, self.season)
            return game

        if (len(self.s_dict) == 0) or (len(slots) == 0):
            raise RuntimeError('''
                    Please check to see that your submission file and
                    tournament files have both have the appropriate season.
                    ''')
        self.games = slots.apply(game_init, axis=1)

    def simulate_games(self, style):
        '''
        This function uses each game class from a specific round
        to predict the winner (set to either the favorite or
        a random sample based on the odds). The winner is added to
        the results file under the appropriate tournament slot.
        To advance teams to the next round us advance_teams()
        afer this or run simulate_round(), which runs both.
        '''

        print(f'Simulating round {self.current_r}...')

        # function pull predicted result from game if in same round
        def find_winner(x):
            if x.r == self.current_r:
                print(x)
                win_id = x.get_winner(self.submission, style)
                if x.strong_team.id == win_id:
                    self.results.update({x.slot: x.strong_team})
                    print(f'{x.strong_team.name} wins!\n')
                elif x.weak_team.id == win_id:
                    self.results.update({x.slot: x.weak_team})
                    print(f'{x.weak_team.name} wins!\n')
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
        if seed is None and style == 'random':
            print('running with no seed')
        else:
            print(f'''
                  Running using seed = {seed}...
                  BEWARE if running multiple simulations!
                  ''')
            np.random.seed(seed)  # seed np at tournament level

        # Run simulations for round 0->6
        while self.current_r < 7:
            self.simulate_round(style)
            self.current_r += 1  # increments round by 1
        print('Tournament complete')

    def get_losses(self, submission):
        '''
        gets losses for all predictions based on the results
        dictionary
        '''

        def logloss(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = submission.get_pred(game_id)
            logloss = pred.logloss[w_id]
            return logloss

        losses = self.games.apply(lambda x: logloss(x))
        return losses

    def get_odds(self, submission):
        '''
        gets odds for all predictions based on the results
        dictionary
        '''

        def calc_odds(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = submission.get_pred(game_id)
            proba = pred.proba[w_id]
            return proba

        odds = self.games.apply(lambda x: calc_odds(x))
        return odds

    def summarize_results(self, summary_dict):
        '''
        Puts the tournament results in a summary format keyed
        by team and round that can be aggregated over multiple
        simulations. This results dict can be made into a pandas
        dataframe by simply calling pd.DataFrame(results) on
        a results dictionary that holds simulated outputs.
        '''

        for slot, team in self.results.items():
            team = team.id
            r = slot[:2]
            if 'R' not in r:
                r = 'R0'

            if summary_dict.get(r) is None:
                summary_dict.update({r: {team: 1}})
            elif summary_dict[r].get(team) is None:
                summary_dict[r].update({team: 1})
            else:
                summary_dict[r][team] += 1

        return summary_dict


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

        self.game_id = '_'.join([str(self.season),
                                 str(self.strong_team.id),
                                 str(self.weak_team.id)])

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
