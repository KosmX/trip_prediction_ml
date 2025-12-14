from abc import ABC

import data
import pandas as pd
from model import Model
from sklearn.metrics import mean_squared_error
import numpy as np

class BaselineModel(Model):
    def __init__(self):
        self.avg_durations = None

    def train(self, df: pd.DataFrame, test: pd.DataFrame = None):
        # calculate average duration for each from-to pair
        # Group by from-to pairs and compute average duration
        self.avg_durations = (
            df
            .groupby(['from', 'to'], as_index=False)['duration']
            .mean()
            .rename(columns={'duration': 'avg_duration'})
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        # merge input data with avg_durations to get predictions
        predictions = pd.merge(
            data,
            self.avg_durations,
            on=['from', 'to'],
            how='left'
        )
        # rename avg_duration to predicted_duration
        predictions = predictions.rename(columns={'avg_duration': 'predicted_duration'})

        return predictions[['from', 'to', 'predicted_duration']]


if __name__ == '__main__':
    # heck, hope Python is able to handle this mess
    # from,to,duration,vehicle_model,vehicle_type,time
    df = data.DataLoader("/app/data").all().sample(frac=1).reset_index(drop=True).copy()

    model = BaselineModel()
    model.train(df)
    predictions = model.predict(df)

    # evaluate with RMSE
    rmse = np.sqrt(mean_squared_error(df['duration'], predictions['predicted_duration']))
    print(f"RMSE: {rmse}")


    exit(0) # other stuff is comment
    # Ensure duration is numeric (coerce errors to NaN)
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Group by from-to pairs and compute average duration
    avg_durations = (
        df
        .groupby(['from', 'to'], as_index=False)['duration']
        .mean()
        .rename(columns={'duration': 'avg_duration'})
    )

    # Sort, display a sample, and persist the results
    avg_durations = avg_durations.sort_values(['from', 'to']).reset_index(drop=True)
    print(avg_durations.head(50))

