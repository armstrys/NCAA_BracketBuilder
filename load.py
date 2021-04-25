from pathlib import Path
import pandas as pd


class Load:
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
