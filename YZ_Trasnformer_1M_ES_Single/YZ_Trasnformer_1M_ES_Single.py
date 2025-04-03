import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import product
import re
import time

FILE_PATH = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)

#insert gap to make sure all the sequences have the same length
def insert_gaps(sequence, mutation_id):
    positions_to_delete = sorted([int(pos[1:]) for pos in re.findall(r'd\d+', mutation_id)])
    mask = np.ones(len(sequence) + len(positions_to_delete), dtype=np.int32)

    sequence = list(sequence)
    offset = 0
    for pos in positions_to_delete:
        adj_pos = pos - 1 + offset
        if adj_pos < len(sequence) + offset:
            sequence.insert(adj_pos, "-")
            mask[adj_pos] = 0
            offset += 1
    
    return "".join(sequence), mask

# ✅ **Tri-nucleotide encoding with gap handling**
def tri_nucleotide_encode(sequence, mask, alphabet="ATGC"):
    triplet_dict = {"".join(kmer): i + 1 for i, kmer in enumerate(product(alphabet, repeat=3))}
    num_vocab = len(triplet_dict) + 1 
    encoded_seq = np.zeros(len(sequence) - 2, dtype=np.int32)
    triplet_mask = np.ones(len(sequence) - 2, dtype=np.int32)
    for i in range(len(sequence) - 2):
        if mask[i] == 0 or mask[i+1] == 0 or mask[i+2] == 0:
            triplet_mask[i] = 0
    
    for i in range(len(sequence) - 2):
        if triplet_mask[i] == 1:  # 只有当三核苷酸中没有gap时才编码
            triplet = sequence[i:i+3]
            if "-" not in triplet:  # 额外检查，确保没有gap
                encoded_seq[i] = triplet_dict.get(triplet, 0)
    
    return encoded_seq, num_vocab, triplet_mask

def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    required_cols = ["id", "full_sequence", "0mM_fracBound", "1mM_fracBound"]
    mutation_ids = df["id"].values
    sequences = df["full_sequence"].values
    y_bound = df["0mM_fracBound"].values
    y_unbound = df["1mM_fracBound"].values

    valid_indices = ~np.isnan(y_bound) & ~np.isnan(y_unbound)
    mutation_ids, sequences, y_bound, y_unbound = mutation_ids[valid_indices], sequences[valid_indices], y_bound[valid_indices], y_unbound[valid_indices]
    aligned_sequences = []
    masks = []
    for seq, mut_id in zip(sequences, mutation_ids):
        aligned_seq, mask = insert_gaps(seq, mut_id)
        aligned_sequences.append(aligned_seq)
        masks.append(mask)
    X_encoded = []
    masks_encoded = []
    _, num_vocab, _ = tri_nucleotide_encode(aligned_sequences[0], masks[0])
    
    for i, (seq, mask) in enumerate(zip(aligned_sequences, masks)):
        if i % 1000 == 0 or i == len(aligned_sequences) - 1:
            print(f"{i+1}/{len(aligned_sequences)}")
        
        encoded_seq, _, triplet_mask = tri_nucleotide_encode(seq, mask)
        X_encoded.append(encoded_seq)
        masks_encoded.append(triplet_mask)

    X = np.array(X_encoded)
    masks_array = np.array(masks_encoded)
    return X, masks_array, y_bound, y_unbound, mutation_ids, sequences, df.iloc[valid_indices], num_vocab

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2, output_dim=1, num_vocab=125):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embbeding dimension ({embed_dim}) has to be divided by({num_heads})"

        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=embed_dim, padding_idx=0)
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

    def forward(self, x, mask=None):
        if not hasattr(self, '_index_checked'):
            if not (x.min() >= 0 and x.max() < self.embedding.num_embeddings):
                x = torch.clamp(x, 0, self.embedding.num_embeddings-1)
            self._index_checked = True
    
        x = self.embedding(x)
        
        x = x + self.pos_embedding[:, :x.shape[1], :]
        if mask is not None:
            attn_mask = mask == 0
            x = self.transformer(x, src_key_padding_mask=attn_mask)
        else:
            x = self.transformer(x)
        
        if mask is not None:
            extended_mask = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = (x * (1 - extended_mask.float())).sum(dim=1) / ((1 - extended_mask.float()).sum(dim=1) + 1e-10)
        else:
            x = torch.mean(x, dim=1)
        return self.fc(x)

def train_model(X_train, masks_train, y_train, X_val, masks_val, y_val, device, model_path, num_vocab, epochs=100, lr=1e-3, batch_size=32):
    
    train_dataset = TensorDataset(X_train, masks_train, y_train)
    val_dataset = TensorDataset(X_val, masks_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    input_dim = X_train.shape[1]
    
    model = TransformerModel(
        input_dim=input_dim,
        embed_dim=64, 
        num_heads=4,  
        num_vocab=num_vocab
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_r2 = -float("inf")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        model.train()
        total_loss = 0
        for i, (batch_X, batch_mask, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(batch_X, batch_mask).squeeze()
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        model.eval()
        all_preds = []
        all_true = []
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_mask, batch_y in val_loader:
                preds = model(batch_X, batch_mask).squeeze()
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_true.append(batch_y.cpu().numpy())
            
            try:
                test_preds = np.concatenate(all_preds)
                test_true = np.concatenate(all_true)
                r2 = r2_score(test_true, test_preds)
            except:
                r2 = -1.0

        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} finished (time {epoch_time:.1f}sec)")
        print(f"total loss: {total_loss/len(train_loader):.4f} | validate loss: {val_loss/len(val_loader):.4f} | R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), model_path)
            print(f"(R²: {r2:.4f})")

    print(f"\ntraining finished, best R²: {best_r2:.4f} | and model saved to {model_path}")
    return model

# ✅ main
if __name__ == "__main__":
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = False  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        torch.cuda.empty_cache()

    try:
        X, masks, y_bound, y_unbound, mutation_ids, sequences, df_filtered, num_vocab = preprocess_data(FILE_PATH)

        X_train, X_test, masks_train, masks_test, y_bound_train, y_bound_test, y_unbound_train, y_unbound_test = train_test_split(
            X, masks, y_bound, y_unbound, test_size=0.1, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.long).to(device)
        X_test = torch.tensor(X_test, dtype=torch.long).to(device)
        
        masks_train = torch.tensor(masks_train, dtype=torch.bool).to(device)
        masks_test = torch.tensor(masks_test, dtype=torch.bool).to(device)

        y_bound_train = torch.tensor(y_bound_train, dtype=torch.float32).to(device)
        y_bound_test = torch.tensor(y_bound_test, dtype=torch.float32).to(device)

        y_unbound_train = torch.tensor(y_unbound_train, dtype=torch.float32).to(device)
        y_unbound_test = torch.tensor(y_unbound_test, dtype=torch.float32).to(device)

        model_path_bound = os.path.join(SAVE_PATH, "transformer_model_bound.pt")
        train_model(X_train, masks_train, y_bound_train, X_test, masks_test, y_bound_test, device, model_path_bound, num_vocab)

        model_path_unbound = os.path.join(SAVE_PATH, "transformer_model_unbound.pt")
        train_model(X_train, masks_train, y_unbound_train, X_test, masks_test, y_unbound_test, device, model_path_unbound, num_vocab)

        print("✅ training finished, model have been saved！")
    
    except Exception as e:
        import traceback
        print(f"❌ error: {str(e)}")
        print(traceback.format_exc())