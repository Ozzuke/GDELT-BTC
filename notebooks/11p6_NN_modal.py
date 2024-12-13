import modal

stub = modal.App("btc-price-prediction")

# create image
image = modal.Image.debian_slim().pip_install(
    "torch",
    "pandas",
    "numpy",
    "scikit-learn",
    "joblib",
    "pyarrow"
)

vol = modal.Volume.lookup('gdelt-btc')


@stub.function(
    image=image,
    gpu="T4",
    timeout=3600,
    memory=16384,  # 16GB RAM
    volumes={"/data": vol}
)
def train_model():
    import pandas as pd
    import joblib
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    class GdeltBtcDataset(Dataset):
        def __init__(self, data, seq_len=4):
            self.data = torch.FloatTensor(data.values)
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            X = self.data[idx:idx + self.seq_len, :-1]
            y = self.data[idx + self.seq_len, -1]
            return X, y

    class BTCPredictor(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )

            self.bn = nn.BatchNorm1d(hidden_size)

            self.regresor = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(16, 1)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_timestep = lstm_out[:, -1, :]
            norm = self.bn(last_timestep)
            return self.regresor(norm)

    def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda', patience=10):
        model = model.to(device)
        best_val_loss = np.inf
        patience_counter = 0
        training_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(X_batch)
                    loss = criterion(y_pred.squeeze(), y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        return training_losses, val_losses

    # load data
    df = pd.read_parquet('/data/prepd_99q.parquet')
    scaler = joblib.load('/data/scaler_99q.joblib')
    encoders = joblib.load('/data/encoders_99q.joblib')

    # training setup
    input_size = df.shape[1] - 1
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    seq_len = 6
    lr = 5e-3
    batch_size = 32
    num_epochs = 100
    patience = 30

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(df) // 5)
    for train_idx, val_idx in tscv.split(df):
        pass
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]

    train_dataset = GdeltBtcDataset(train_data, seq_len=seq_len)
    val_dataset = GdeltBtcDataset(val_data, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BTCPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    training_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device='cuda',
        patience=patience
    )

    # Save model
    torch.save(model.state_dict(), '/data/model.pth')
    return training_losses, val_losses


@stub.local_entrypoint()
def main():
    training_losses, val_losses = train_model.remote()
    print("Training completed!")

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(training_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.show()
