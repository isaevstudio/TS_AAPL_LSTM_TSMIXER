import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import optuna

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset Class for Time Series
class TimeSeriesDataset(Dataset):

    print('CLASS: TimeSeriesDataset')

    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length, :-1]  # Features
        target = self.data[idx + self.sequence_length, -1]  # Target
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# Define the LSTM Model
class LSTMModel(nn.Module):

    print('CLASS: LSTMModel')

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Output from the last time step
        return out


# Prepare the Data
def prepare_data(df, sequence_length, test_size=0.2):
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Split data into train and test
    train_data, test_data = train_test_split(scaled_data, test_size=test_size, shuffle=False)

    # Convert to PyTorch datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)

    return train_dataset, test_dataset, scaler


# Train the Model
def train_model(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for sequences, targets in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


# Make Predictions
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    y_test = []
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
            y_test.extend(targets.numpy())
    return np.array(predictions).flatten(), np.array(y_test).flatten()


# ------------------------------OPTUNA

# Prepare Data
def prepare_data(df, sequence_length, target_column='close'):
    split_idx = int(len(df) * 0.8)
    train_data = df[:split_idx]
    test_data = df[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    def create_sequences(data, seq_len, target_idx):
        x, y = [], []
        for i in range(len(data) - seq_len):
            x.append(data[i:i + seq_len, :-1])  # All features except target
            y.append(data[i + seq_len, target_idx])  # Target column
        return np.array(x), np.array(y)

    target_idx = list(df.columns).index(target_column)
    x_train, y_train = create_sequences(train_scaled, sequence_length, target_idx)
    x_test, y_test = create_sequences(test_scaled, sequence_length, target_idx)

    return x_train, y_train, x_test, y_test, scaler

# Objective Function for Optuna
def objective(trial, df):
    # Hyperparameter Tuning
    sequence_length = trial.suggest_int('sequence_length', 30, 100)
    neurons_lstm = trial.suggest_int('neurons_lstm', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Prepare data
    x_train, y_train, x_test, y_test, scaler = prepare_data(df, sequence_length, target_column='close')
    input_dim = x_train.shape[2]

    # Build model
    model = Sequential()
    model.add(LSTM(neurons_lstm, activation='tanh', input_shape=(sequence_length, input_dim), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Train model
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=20,  # Fixed number of epochs for tuning
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    )

    # Evaluate model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate([np.zeros((predictions.shape[0], df.shape[1] - 1)), predictions], axis=1))[:, -1]
    y_test = scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], df.shape[1] - 1)), y_test.reshape(-1, 1)], axis=1))[:, -1]

    return np.array(predictions).flatten(), np.array(y_test).flatten()

# Optuna Optimization
def tune_hyperparameters(df):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, df), n_trials=50)

    print("Best Hyperparameters:")
    print(study.best_params)

    return study.best_params