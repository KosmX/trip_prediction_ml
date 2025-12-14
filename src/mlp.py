import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import Model
import numpy as np
import math

# stop_id,stop_name,stop_lat,stop_lon,stop_code,location_type,location_sub_type,parent_station,wheelchair_boarding
stops = pd.read_csv("/app/data/stops.txt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


def sinusoidal_positional_encoding(value: float, embed_dim: int) -> np.ndarray:
    """
    Generates sinusoidal positional embeddings for a single scalar value.
    Args:
        value (float): The scalar value (e.g., latitude or longitude).
        embed_dim (int): The desired embedding dimension (must be even).
    Returns:
        np.ndarray: A 1D numpy array of shape (embed_dim,) containing the embedding.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be an even number for sinusoidal positional encoding.")

    position = np.array(value)
    # Using a base of 10000, typical for transformers
    div_term = np.exp(np.arange(0, embed_dim, 2) * (math.log(10000.0) / embed_dim))

    pe = np.zeros(embed_dim)
    pe[0::2] = np.sin(position * div_term * 20)  # budapest all
    pe[1::2] = np.cos(position * div_term * 20)  # budapest all
    return pe


def create_embeds(df: pd.DataFrame) -> dict:
    """
    I have no fricking idea how does this function work, but somehow it does.
    :param df: times
    :return: something
    """
    # 1. Stop Embeddings (Latitude and Longitude)
    unique_stop_ids = df[['from', 'to']].drop_duplicates()

    # from: same index as unique_stop_ids, looked up 'from' from stops dataframe
    from_stuff = unique_stop_ids.merge(stops, left_on=['from'], right_on=['stop_id'])
    to_stuff = unique_stop_ids.merge(stops, left_on=['to'], right_on=['stop_id'])

    unique_stop_ids['lat'] = ((from_stuff['stop_lat'] + to_stuff['stop_lat']) / 2).map(
        lambda x: sinusoidal_positional_encoding(x, 12))
    unique_stop_ids['lon'] = ((from_stuff['stop_lon'] + to_stuff['stop_lon']) / 2).map(
        lambda x: sinusoidal_positional_encoding(x, 12))

    # 2. Vehicle Type Embeddings (4 bits)
    unique_vehicle_types = df['vehicle_type'].astype(int).unique()

    # Generate a unique 4-dimensional vector for each vehicle type.
    # Using a fixed seed and the vehicle type itself to ensure deterministic and unique embeddings.
    np.random.seed(42)  # For reproducibility
    vehicle_embedding_dict = {}
    for vt in unique_vehicle_types:
        rng = np.random.default_rng(42 + vt)  # Use vehicle type in seed for uniqueness
        vehicle_embedding_dict[vt] = rng.uniform(-1, 1, 4)  # Random 4-dim vector

    vehicle_embedding_features = pd.DataFrame.from_dict(
        vehicle_embedding_dict,
        orient='index',
        columns=[f'vehicle_embed_{i}' for i in range(4)]
    )
    vehicle_embedding_features.index.name = 'vehicle_type'

    # Define embed_dim based on the expected output of sinusoidal_positional_encoding (from bv11y8jw0uGI)
    embed_dim = 12

    for i in range(embed_dim):
        unique_stop_ids['lat_embed_{}'.format(i)] = unique_stop_ids['lat'].map(
            lambda x: x[i] if type(x) is not float else 0)
        unique_stop_ids['lon_embed_{}'.format(i)] = unique_stop_ids['lon'].map(
            lambda x: x[i] if type(x) is not float else 0)

    return {
        'stop_embed': unique_stop_ids.drop('lat', axis=1).drop('lon', axis=1),
        'vehicle_embed': vehicle_embedding_features
    }


import data

def prepare_data(df: pd.DataFrame, embeddings_dict):
    """
    Merges the main dataframe with the embeddings dictionaries
    and processes the time column.
    """
    # 1. Merge Stop Embeddings (Lat/Lon)
    # The stop_embed df in your code is keyed by ['from', 'to']
    stop_embeds = embeddings_dict['stop_embed']
    df_merged = df.merge(stop_embeds, on=['from', 'to'], how='left')

    # 2. Merge Vehicle Embeddings
    # The vehicle_embed df is keyed by vehicle_type index
    veh_embeds = embeddings_dict['vehicle_embed']
    df_merged = df_merged.merge(veh_embeds, left_on='vehicle_type', right_index=True, how='left')

    # 3. Process Time (Cyclic Encoding)
    # Neural nets struggle with raw timestamps. We convert to Hour/Day cyclic features.
    df_merged['time'] = pd.to_datetime(df_merged['time'])

    # Hour of day (0-23) mapped to unit circle (sin/cos)
    df_merged['hour_sin'] = np.sin(2 * np.pi * df_merged['time'].dt.hour / 24)
    df_merged['hour_cos'] = np.cos(2 * np.pi * df_merged['time'].dt.hour / 24)

    # Day of week (0-6)
    df_merged['day_sin'] = np.sin(2 * np.pi * df_merged['time'].dt.dayofweek / 7)
    df_merged['day_cos'] = np.cos(2 * np.pi * df_merged['time'].dt.dayofweek / 7)
    return df_merged

def prepare_training_data(df: pd.DataFrame, embeddings_dict):
    df_merged = prepare_data(df, embeddings_dict)

    # 4. Select Feature Columns
    # Identify embedding columns dynamically
    lat_cols = [c for c in df_merged.columns if 'lat_embed_' in c]
    lon_cols = [c for c in df_merged.columns if 'lon_embed_' in c]
    veh_cols = [c for c in df_merged.columns if 'vehicle_embed_' in c]
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    feature_cols = lat_cols + lon_cols + veh_cols + time_cols

    # Extract X (features) and y (target)
    X = df_merged[feature_cols].values.astype(np.float32)
    y = df_merged['duration'].values.astype(np.float32).reshape(-1, 1)

    return X, y, len(feature_cols)

def preare_prediction_data(df: pd.DataFrame, embeddings_dict):
    df_merged = prepare_data(df, embeddings_dict)

    # 4. Select Feature Columns
    # Identify embedding columns dynamically
    lat_cols = [c for c in df_merged.columns if 'lat_embed_' in c]
    lon_cols = [c for c in df_merged.columns if 'lon_embed_' in c]
    veh_cols = [c for c in df_merged.columns if 'vehicle_embed_' in c]
    time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    feature_cols = lat_cols + lon_cols + veh_cols + time_cols

    # Extract X (features)
    X = df_merged[feature_cols].values.astype(np.float32)

    return X, len(feature_cols)


# Define a standard PyTorch Dataset
class TripDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DurationPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DurationPredictor, self).__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output Layer (Scalar regression)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class MLPModel(Model):
    def __init__(self):
        self.model = None
        self.dict = None

    def train(self, df: pd.DataFrame, test: pd.DataFrame):
        df = df.sample(frac=1).reset_index(drop=True).copy()
        embedding_dict = create_embeds(df)
        self.dict = embedding_dict

        X_data, y_data, input_dim = prepare_training_data(df, embedding_dict)

        dataset = TripDataset(X_data, y_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        model = DurationPredictor(input_dim=input_dim).to(device)
        self.model = model
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        test_data, _ = preare_prediction_data(test.drop(columns='duration'), self.dict)
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

        # --- 4. TRAINING LOOP ---
        print(f"Starting training with input dimension: {input_dim}")

        epochs = 40 # ??? probably waaaaaaay more

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Zero gradients
                optimizer.zero_grad()
                X = batch_X.to(device)
                y = batch_y.to(device)

                # Forward pass
                predictions = model(X)

                # Calculate loss
                loss = criterion(predictions, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
                # RMSE
                with torch.no_grad():
                    model.eval()
                    test_predictions = model(test_tensor)
                    rmse = torch.sqrt(criterion(test_predictions, torch.tensor(test['duration'].values.astype(np.float32).reshape(-1, 1)).to(device)))
                    print(f"Test RMSE after epoch {epoch + 1}: {rmse.item():.4f}")
                    model.train()


    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        model: DurationPredictor|None = self.model
        if model is None:
            raise RuntimeError("Model has not been trained yet.")

        X_data, input_dim = preare_prediction_data(df, self.dict)

        input_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(input_tensor).to('cpu')
            # Squeeze to get rid of the extra dimensions, and convert to standard Python float
            return prediction.squeeze().item()



if __name__ == '__main__':
    data = data.DataLoader("/app/data").all()
    embeds = create_embeds(data)
    print(embeds)
    prep = prepare_training_data(data, embeds)
    print(prep)
