from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(self, data, test):
        pass

    @abstractmethod
    def predict(self, data) -> pd.DataFrame:
        pass

    def predict_on(self, prev_stop: str, next_stop: str, current_time: str, vehicle_type: int) -> float:
        input_data = {
            'from': [prev_stop],
            'to': [next_stop],
            'vehicle_type': [vehicle_type],
            'time': [current_time]
        }
        input_df = pd.DataFrame(input_data)
        prediction = self.predict(input_df)
        return prediction.iloc[0]
