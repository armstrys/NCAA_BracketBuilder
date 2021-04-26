import re
import numpy as np


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
            np.random.seed(seed)  # seed np at tournament level

        # Run simulations for round 0->6
        while self.current_r < 7:
            self.simulate_round(style)
            self.current_r += 1  # increments round by 1
        print('Tournament complete')


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
        if self.strong_team is None or self.weak_team is None:
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
        if self.strong_team is None or self.weak_team is None:
            return True
        else:
            return False

    def get_winner(self, submission, style, seed=0):
        if self.strong_team is None or self.weak_team is None:
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


class Team:
    def __init__(self, t_id, name, seed):
        self.id = t_id
        self.name = name
        self.seed = seed

    def __repr__(self):
        return f'{self.seed} {self.name} - TeamID: {self.id}'
