from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(self, data, test):
        pass

    @abstractmethod
    def predict(self, data) -> pd.DataFrame:
        pass

    def predict_on(self, prev_stop: str, next_stop: str, current_time: pd.Timestamp, route_type: int, vehicle_type: int = 0) -> float:
        input_data = {
            'prev_stop': [prev_stop],
            'next_stop': [next_stop],
            'current_time': [current_time],
            'route_type': [route_type],
            'vehicle_type': [vehicle_type]
        }
        input_df = pd.DataFrame(input_data)
        prediction = self.predict(input_df)
        return prediction.iloc[0]
