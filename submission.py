import numpy as np


class Submission:
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
