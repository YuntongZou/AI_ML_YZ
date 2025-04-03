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
from itertools import product

file_path = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ **Tri-nucleotide encoding**
def tri_nucleotide_encode(sequence, alphabet="ATGC"):
    triplet_dict = {"".join(kmer): i for i, kmer in enumerate(product(alphabet, repeat=3))}
    encoded_seq = np.zeros(len(sequence) - 2, dtype=np.int32)  
    for i in range(len(sequence) - 2):
        triplet = sequence[i : i + 3]
        encoded_seq[i] = triplet_dict.get(triplet, 0)  
    return encoded_seq

# ✅ preprocess_Data
def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    sequences = df["Mutated_Sequence"].values
    y_values = df.iloc[:, 1:7].values  # (N, 6)

    valid_indices = ~np.isnan(y_values).any(axis=1)
    sequences = sequences[valid_indices]
    y_values = y_values[valid_indices]

    X = np.array([tri_nucleotide_encode(seq) for seq in sequences])

    return X, y_values, df


class IntegratedTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, output_dim=6):  
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=64, embedding_dim=embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)  

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  
        return self.fc(x)  

def train_single_model(X_train, y_train, X_test, y_test, device, model_path, epochs=100, lr=1e-3, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = IntegratedTransformer(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_r2 = -float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)  

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).cpu().numpy()
            test_true = y_test.cpu().numpy()
            r2_scores = [r2_score(test_true[:, i], test_preds[:, i]) for i in range(6)]
            avg_r2 = np.mean(r2_scores)

        print(f"📌 Epoch {epoch+1} | Loss: {total_loss:.4f} | Avg R²: {avg_r2:.4f}")

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            torch.save(model.state_dict(), model_path)

    print(f"\n✅ traning finished R²: {best_r2:.4f} | model has been saved to {model_path}")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    X, y_values, df = preprocess_data(file_path)
    
   
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y_values, np.arange(len(X)), test_size=0.2, random_state=42
    )

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    
    model_path = os.path.join(SAVE_PATH, "integrated_transformer_model.pt")
    model = IntegratedTransformer(input_dim=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_predictions_df = pd.DataFrame({"Mutated_Sequence": df["Mutated_Sequence"].iloc[test_indices]})
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    for i in range(6):
        test_predictions_df[f"True_T{i+1}"] = y_test.cpu().numpy()[:, i]
        test_predictions_df[f"Pred_T{i+1}"] = y_pred[:, i]

    test_predictions_df.to_csv(os.path.join(SAVE_PATH, "integrated_transformer_predictions.csv"), index=False)

    plt.figure(figsize=(6, 6))
    for i in range(6):
        plt.scatter(test_predictions_df[f"True_T{i+1}"], test_predictions_df[f"Pred_T{i+1}"], alpha=0.5, label=f"T{i+1}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.legend()
    plt.savefig(os.path.join(SAVE_PATH, "transformer_r2_scatter_test.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sample_indices = np.random.choice(len(test_predictions_df), size=5, replace=False)

    for i, idx in enumerate(sample_indices):
        plt.plot(range(1, 7), test_predictions_df.iloc[idx, 1:7].values, "o--", label=f"True Seq {i+1}", alpha=0.7)
        plt.plot(range(1, 7), test_predictions_df.iloc[idx, 7:].values, "s-", label=f"Pred Seq {i+1}", alpha=0.7)

    plt.xlabel("Concentration Index")
    plt.ylabel("Measured Value")
    plt.title("Predicted vs True Curves for 5 Test Sequences")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "transformer_curve_comparison_test.png"), dpi=300)
    plt.close()

    print("✅ the predcition results has beensaved to CSV file and all the figures have been saved！")
