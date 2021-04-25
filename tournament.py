import re
import numpy as np


class Tournament:

    def __init__(self, files, submission, season):
        self.submission = submission
        self.mw = files.mw
        self.season = season
        self.current_r = 0
        self.results = {}

        seeds = files.seeds[files.seeds['Season'] == self.season]
        self.s_dict = (seeds.set_index('Seed')['TeamID']
                            .to_dict())
        self.s_dict_rev = (seeds.set_index('TeamID')['Seed']
                                .to_dict())

        # Setting up data frame to track games
        if self.mw == 'M':
            slots = files.slots[files.slots['Season'] == season].copy()
            slots.drop(columns='Season', inplace=True)
        else:
            slots = files.slots

        def game_init(row):
            game = Game(row, submission, files.t_dict,
                        self.s_dict, self.season)
            return game

        self.games = slots.apply(game_init, axis=1)

    def simulate_round(self, style):
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

        self.games.apply(find_winner)
        self.games.apply(lambda x: x.add_teams(self.results))

        self.current_r += 1

    def simulate_tournament(self, style, seed=None):
        if seed is None:
            print('running with no seed')
        else:
            np.random.seed(seed)

        while self.current_r < 7:
            print(f'Simulating round {self.current_r}...')
            self.simulate_round(style)

        print('Done')


class Game:

    def __init__(self, row_slots, submission, t_dict, s_dict, season):
        self.season = season
        self.slot = row_slots['Slot']
        self.strong_seed = row_slots['StrongSeed']
        self.weak_seed = row_slots['WeakSeed']

        r = re.compile(r'(R.)[WXYZC].')
        match = r.search(self.slot)
        if match is not None:
            self.r_label = match.group(1)
        else:
            self.r_label = 'R0'

        if self.r_label is None:
            self.r_label = 'R0'
        self.r = int(self.r_label[-1])

        self.strong_team = None
        self.weak_team = None
        strong_id = s_dict.get(self.strong_seed)
        weak_id = s_dict.get(self.weak_seed)

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

        game_id = '_'.join([str(self.season),
                            str(self.strong_team.id),
                            str(self.weak_team.id)])

        if style == 'chalk':
            win_id = (
                submission.get_pred(game_id)
                          .get_favored()
                          )
            return win_id
        elif style == 'random':
            win_id = (
                submission.get_pred(game_id)
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
