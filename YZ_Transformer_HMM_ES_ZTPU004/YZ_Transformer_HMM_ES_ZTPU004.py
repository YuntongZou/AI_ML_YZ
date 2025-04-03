import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file_path = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    mutation_sites = df["mutation_site"]
    time_series = df.iloc[:, 1:7].values  # (N, 6)

    # One-hot 
    nucleotide_map = {"A": 0, "T": 1, "G": 2, "C": 3}
    num_nt = len(nucleotide_map)
    
    encoded_mutations = np.array([[nucleotide_map[nt] for nt in seq] for seq in mutation_sites])
    one_hot_mutations = np.eye(num_nt)[encoded_mutations]  # (N, sequence_length, 4)

    # training HMM
    hmm_model = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=1000)
    hmm_model.fit(time_series)

    # predict HMM states for each time point
    hmm_states_seq = np.zeros_like(time_series, dtype=int)  # 预分配 (N, 6)
    for t in range(time_series.shape[1]):  # 遍历6个时间点
        hmm_states_seq[:, t] = hmm_model.predict(time_series[:, t].reshape(-1, 1))

    print(f"✅ One-hot Mutations Shape: {one_hot_mutations.shape}")  # (N, sequence_length, 4)
    print(f"✅ HMM States Shape: {hmm_states_seq.shape}")  # (N, 6)
    print(f"✅ Time Series Shape: {time_series.shape}")  # (N, 6)

    return one_hot_mutations, hmm_states_seq, time_series, df

class HMMTransformer(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        self.fc_nt = nn.Linear(input_dim, hidden_dim)
        self.hmm_embed = nn.Embedding(7, hidden_dim)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, 6)

    def forward(self, nt_input, hmm_input):
        nt_emb = self.fc_nt(nt_input)
        hmm_emb = self.hmm_embed(hmm_input)

        Q = self.W_Q(nt_emb)
        K = self.W_K(hmm_emb)
        V = self.W_V(hmm_emb)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(64)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)

        combined = nt_emb + context
        out = self.transformer(combined)
        out = torch.mean(out, dim=1)
        return self.fc(out)

def train_model(X_nt, X_hmm, y, device):
    # ✅ add new index array for test set
    indices = np.arange(len(X_nt))
    
    (X_nt_train, X_nt_test, 
     X_hmm_train, X_hmm_test, 
     y_train, y_test, 
     idx_train, idx_test) = train_test_split(
        X_nt, X_hmm, y, indices, 
        test_size=0.2, 
        random_state=42
    )

    print(f"✅ Train HMM Shape: {X_hmm_train.shape}")
    print(f"✅ Test HMM Shape: {X_hmm_test.shape}")

    train_nt = torch.tensor(X_nt_train, dtype=torch.float32).to(device)
    train_hmm = torch.tensor(X_hmm_train, dtype=torch.long).to(device)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)

    test_nt = torch.tensor(X_nt_test, dtype=torch.float32).to(device)
    test_hmm = torch.tensor(X_hmm_test, dtype=torch.long).to(device)
    test_y = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = HMMTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n🚀 Training starts...\n")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        preds = model(train_nt, train_hmm)
        loss = criterion(preds, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"📌 Epoch {epoch + 1} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(test_nt, test_hmm).cpu().numpy()

    mse = np.mean((test_preds - y_test) ** 2)
    r2 = r2_score(y_test.flatten(), test_preds.flatten())
    print(f"\n✅ Test MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return test_preds, y_test, idx_test

def visualize_results(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(y_true[i], '--', label=f"True Sample {i+1}")
        plt.plot(y_pred[i], '-', label=f"Pred Sample {i+1}")
    plt.legend()
    plt.xlabel("Time Points")
    plt.ylabel("Value")
    plt.title("Time Series Predictions")
    plt.savefig(f"{save_path}/predictions_2.png", dpi=300)

    plt.figure(figsize=(8, 6))
    sns.histplot((y_true - y_pred).flatten(), kde=True, bins=30)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.savefig(f"{save_path}/error_distribution_2.png", dpi=300)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true.flatten(), y=y_pred.flatten(), alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    plt.text(y_true.min(), y_true.max(), f"$R^2$ = {r2:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs. Actual Values")
    plt.savefig(f"{save_path}/correlation_scatter_2.png", dpi=300)

def plot_fold_change_r2(y_true, y_pred, save_path):
    fc_true = y_true[:, -1] / y_true[:, 0]
    fc_pred = y_pred[:, -1] / y_pred[:, 0]

    r2_fc = r2_score(fc_true, fc_pred)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=fc_true, y=fc_pred, alpha=0.5)
    plt.plot([fc_true.min(), fc_true.max()], [fc_true.min(), fc_true.max()], 'r--')

    plt.text(fc_true.min(), fc_true.max(), f"$R^2$ = {r2_fc:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("True Fold Change")
    plt.ylabel("Predicted Fold Change")
    plt.title("Fold Change R²")
    plt.savefig(f"{save_path}/fold_change_r2.png", dpi=300)

def save_predictions_to_csv(df, test_idx, y_true, y_pred, save_path):
    """ save results to CSV file"""
    test_df = df.iloc[test_idx].copy()  
    
    for i in range(y_true.shape[1]):
        test_df[f"True_T{i+1}"] = y_true[:, i]
        test_df[f"Pred_T{i+1}"] = y_pred[:, i]
    
    test_df["True_FC"] = test_df["True_T6"] / test_df["True_T1"]
    test_df["Pred_FC"] = test_df["Pred_T6"] / test_df["Pred_T1"]
    
    csv_path = os.path.join(save_path, "test_predictions.csv")
    test_df.to_csv(csv_path, index=False)
    print(f"✅ prediction results has been saved to: {csv_path}")

if __name__ == "__main__":
    encoded_nt, hmm_states, time_series, df = load_and_preprocess(file_path)
    
    preds, truths, test_indices = train_model(
        encoded_nt, 
        hmm_states, 
        time_series, 
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    visualize_results(truths, preds, SAVE_PATH)
    plot_fold_change_r2(truths, preds, SAVE_PATH)
    
    save_predictions_to_csv(df, test_indices, truths, preds, SAVE_PATH)