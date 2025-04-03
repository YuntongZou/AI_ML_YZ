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
import re

# ✅ Enable CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# ✅ **Set File Paths**
FILE_PATH = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"✅ Loading data from: {FILE_PATH}")

# ✅ **Insert Gaps Based on ID**
def insert_gaps(sequence, mutation_id):
    positions_to_delete = sorted([int(pos[1:]) for pos in re.findall(r'd\d+', mutation_id)])
    sequence = list(sequence)

    for pos in positions_to_delete:
        if pos - 1 < len(sequence):
            sequence.insert(pos - 1, "-")  # Insert '-'

    return "".join(sequence)

# ✅ **Tri-Nucleotide Encoding (Modified for Gaps)**
def tri_nucleotide_encode(sequence, alphabet="ATGC-"):
    triplet_dict = {"".join(kmer): i+1 for i, kmer in enumerate(product(alphabet.replace('-', ''), repeat=3))}
    triplet_dict["---"] = 0  # Treat gaps as 0

    encoded_seq = np.zeros(len(sequence) - 2, dtype=np.int32)
    for i in range(len(sequence) - 2):
        triplet = sequence[i : i + 3]
        encoded_seq[i] = triplet_dict.get(triplet, 0)  # Default to 0 if missing

    return encoded_seq, len(triplet_dict)

# ✅ **Data Preprocessing**
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    if "id" not in df.columns or "full_sequence" not in df.columns:
        raise ValueError("❌ Missing 'id' or 'full_sequence' column!")

    mutation_ids = df["id"].values
    sequences = df["full_sequence"].values
    y_values = df["fold_change"].values  # Only using fold_change as target

    valid_indices = ~np.isnan(y_values)
    mutation_ids, sequences, y_values = mutation_ids[valid_indices], sequences[valid_indices], y_values[valid_indices]

    # Insert gaps based on mutation ID
    aligned_sequences = [insert_gaps(seq, mut_id) for seq, mut_id in zip(sequences, mutation_ids)]
    
    # Encode sequences
    X_encoded = []
    for seq in aligned_sequences:
        encoded_seq, num_vocab = tri_nucleotide_encode(seq)
        X_encoded.append(encoded_seq)

    X = np.array(X_encoded)
    print(f"✅ Encoded Data Shape: {X.shape}, Num Vocab: {num_vocab}")

    return X, y_values, aligned_sequences, mutation_ids, df.iloc[valid_indices], num_vocab

# ✅ **Transformer Model**
class IntegratedTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, output_dim=1, num_vocab=125):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embed_dim, padding_idx=0)  # Gaps as padding
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=256, 
            dropout=0.1, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        print(f"🚀 Forward pass: Min index = {x.min().item()}, Max index = {x.max().item()}")
        x = self.embedding(x) + self.pos_embedding[:, :x.shape[1], :]
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Mean pooling
        return self.fc(x).squeeze()

def train_model(X_train, y_train, X_test, y_test, device, model_path, num_vocab, epochs=100, lr=1e-3, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = IntegratedTransformer(input_dim=X_train.shape[1], num_vocab=num_vocab).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_r2 = -float("inf")
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    # Print model and data info for debugging
    print(f"✅ Sequence length: {X_train.shape[1]}")
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Effective batch size with accumulation: {batch_size * accumulation_steps}")
    if torch.cuda.is_available():
        print(f"✅ Max GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for step, (batch_X, batch_y) in enumerate(train_loader):
            preds = model(batch_X)
            loss = criterion(preds, batch_y) / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps
            
            if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                print(f"✅ Step {step}/{len(train_loader)}: loss={total_loss/(step%accumulation_steps+1):.6f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_preds = []
            test_targets = []
            for batch_X, batch_y in test_loader:
                batch_preds = model(batch_X)
                test_preds.append(batch_preds.cpu().numpy())
                test_targets.append(batch_y.cpu().numpy())
            
            test_preds = np.concatenate(test_preds)
            test_targets = np.concatenate(test_targets)
            r2 = r2_score(test_targets, test_preds)
            mse = np.mean((test_targets - test_preds) ** 2)
            
            print(f"📌 Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.6f} | Test MSE: {mse:.6f} | Test R2: {r2:.6f}")
            
            if r2 > best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), model_path)
                print(f"✅ Model saved with R2: {r2:.6f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_path))
    return model
# ✅ **Main Execution**
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Running on: {device}")

    X, y_values, sequences, mutation_ids, df_filtered, num_vocab = preprocess_data(FILE_PATH)

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y_values, np.arange(len(X)), test_size=0.1, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.long).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    model_path = os.path.join(SAVE_PATH, "integrated_transformer_model.pt")
    
    if not os.path.exists(model_path):
        train_model(X_train, y_train, X_test, y_test, device, model_path, num_vocab)

    print("✅ Training complete!")
    # ✅ Load model & make predictions
    model = IntegratedTransformer(input_dim=X_test.shape[1], num_vocab=num_vocab).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        true_vals = y_test.cpu().numpy()

    # ✅ Prepare prediction DataFrame
    prediction_df = pd.DataFrame({
        "Mutation_ID": mutation_ids[test_indices],
        "Aligned_Sequence": [sequences[i] for i in test_indices],
        "True_Fold_Change": true_vals,
        "Predicted_Fold_Change": preds
    })

    # ✅ Save to CSV
    prediction_file = os.path.join(SAVE_PATH, "test_predictions.csv")
    prediction_df.to_csv(prediction_file, index=False)
    print(f"✅ Predictions saved to: {prediction_file}")
