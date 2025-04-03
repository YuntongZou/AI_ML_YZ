import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#File path
file_path = "/projects/p32363/AI_ML_Data/ZTP_U004_GreB_LE_dose_curve_combined_with_sequences.csv"
SAVE_PATH = "“save/to/your/path”"
os.makedirs(SAVE_PATH, exist_ok=True)

# One hot encoding
def one_hot_encode(sequence, alphabet="ATGC"):
    mapping = {char: i for i, char in enumerate(alphabet)}
    one_hot = np.zeros((len(sequence), len(alphabet)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot.flatten()

# Pre-process the data
def preprocess_data(filepath):
    df = pd.read_csv(filepath)


    sequences = df["Mutated_Sequence"].values
    y_values = df.iloc[:, 1:7].values  # time series = (N, 6)

    valid_indices = ~np.isnan(y_values).any(axis=1)
    sequences = sequences[valid_indices]
    y_values = y_values[valid_indices]

    X = np.array([one_hot_encode(seq) for seq in sequences])

    return X, y_values, df

# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.2))  
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Train the model 
def train_model(X_train, y_train, X_test, y_test, device, model_save_path, epochs=100, lr=1e-3, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_r2 = -float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).squeeze().cpu().numpy()
            test_true = y_test.cpu().numpy()
            r2 = r2_score(test_true, test_preds)

        print(f"📌 Epoch {epoch+1} | Loss: {total_loss:.4f} | R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), model_save_path)

    print("\n✅ training finished, R²: {:.4f}".format(best_r2))
    return model

# Test the model
def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        y_pred = model(X).squeeze().cpu().numpy()
    return y.cpu().numpy(), y_pred

# R2 scatter plot
def plot_r2_scatter(y_true, y_pred, save_path):

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title(f"Test Set Prediction Results (R²: {r2_score(y_true, y_pred):.4f})")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Prediction")
    plt.legend()
    plt.savefig(os.path.join(save_path, "mlp_r2_scatter_test.png"), dpi=300)
    plt.close()

# curve
def plot_curves(y_true_all, y_pred_all, save_path):
    indices = np.random.choice(len(y_true_all), size=5, replace=False)

    plt.figure(figsize=(8, 6))
    for i, idx in enumerate(indices):
        plt.plot(range(1, 7), y_true_all[idx], "o--", label=f"True Seq {i+1}", alpha=0.7)
        plt.plot(range(1, 7), y_pred_all[idx], "s-", label=f"Pred Seq {i+1}", alpha=0.7)

    plt.xlabel("Concentration Index")
    plt.ylabel("Measured Value")
    plt.title("Test Set: Predicted vs True Curves for 5 Sequences")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "mlp_curve_comparison_test.png"), dpi=300)
    plt.close()

# name
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y_values, df = preprocess_data(file_path)

    test_predictions_df = pd.DataFrame({"Mutated_Sequence": df["Mutated_Sequence"]})
    test_indices_all = np.array([], dtype=int) 
    y_pred_all = np.zeros_like(y_values)

    for i in range(y_values.shape[1]):
        print(f"\n🚀  {i+1}")

        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X, y_values[:, i], np.arange(len(X)), test_size=0.2, random_state=42
        )

        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

        model_save_path = os.path.join(SAVE_PATH, f"mlp_model_T{i+1}.pt")
        model = train_model(X_train, y_train, X_test, y_test, device, model_save_path)

        model.load_state_dict(torch.load(model_save_path, map_location=device))

        y_true, y_pred = evaluate_model(model, X_test, y_test, device)
        y_pred_all[test_indices, i] = y_pred
        test_indices_all = np.concatenate((test_indices_all, test_indices))

        test_predictions_df.loc[test_indices, f"True_T{i+1}"] = y_true
        test_predictions_df.loc[test_indices, f"Pred_T{i+1}"] = y_pred

    test_predictions_df = test_predictions_df.iloc[test_indices_all]

    csv_save_path = os.path.join(SAVE_PATH, "mlp_test_predictions.csv")
    test_predictions_df.to_csv(csv_save_path, index=False)
    print(f"✅ prediction of test data is saved to : {csv_save_path}")

    plot_r2_scatter(test_predictions_df["True_T6"], test_predictions_df["Pred_T6"], SAVE_PATH)
    plot_curves(y_values[test_indices_all], y_pred_all[test_indices_all], SAVE_PATH)
