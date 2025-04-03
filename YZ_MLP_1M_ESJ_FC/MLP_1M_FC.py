import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
import flax.serialization
import re

# ✅ **Set Paths**
FILE_PATH = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ **One-hot encoding function (Including `-` for gaps)**
def one_hot_encode(sequence, alphabet="ATGC-"):
    mapping = {char: i for i, char in enumerate(alphabet)}
    one_hot = np.zeros((len(sequence), len(alphabet)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot.flatten()

# ✅ **Insert Gaps Based on Mutation ID**
def insert_gaps(sequence, mutation_id):
    positions_to_delete = sorted([int(pos[1:]) for pos in re.findall(r'd\d+', mutation_id)])  
    sequence = list(sequence)  
    for pos in positions_to_delete:
        if pos - 1 < len(sequence):
            sequence.insert(pos - 1, "-")  
    return "".join(sequence)

# ✅ **Data Preprocessing Function**
def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    required_cols = ["id", "full_sequence", "0mM_fracBound", "1mM_fracBound", "fold_change"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column '{col}' in dataset!")

    mutation_ids = df["id"].values
    sequences = df["full_sequence"].values
    y_bound = df["0mM_fracBound"].values  
    y_unbound = df["1mM_fracBound"].values  
    y_fold_change = df["fold_change"].values  

    valid_indices = ~np.isnan(y_bound) & ~np.isnan(y_unbound) & ~np.isnan(y_fold_change)
    mutation_ids, sequences, y_bound, y_unbound, y_fold_change = (
        mutation_ids[valid_indices], sequences[valid_indices], 
        y_bound[valid_indices], y_unbound[valid_indices], y_fold_change[valid_indices]
    )

    # Insert gaps based on mutation ID
    aligned_sequences = [insert_gaps(seq, mut_id) for seq, mut_id in zip(sequences, mutation_ids)]

    # One-hot encode sequences
    X = np.array([one_hot_encode(seq) for seq in aligned_sequences])

    # **Stack targets together**: `[0mM_fracBound, 1mM_fracBound, fold_change]`
    Y = np.column_stack([y_bound, y_unbound, y_fold_change])

    return X, Y, aligned_sequences, mutation_ids

# ✅ **Define MLP Model**
class MLP(nn.Module):
    width: int
    depth: int
    
    def setup(self):
        self.layers = [nn.Dense(self.width) for _ in range(self.depth)]
        self.output_layer = nn.Dense(3)  # Output 3 values (0mM, 1mM, fold_change)
    
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return self.output_layer(x)

# ✅ **Training Function**
def train(model, params, data, epochs=100, lr=1e-3, batch_size=32):
    rng = jax.random.PRNGKey(0)
    X, Y = data

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(params, X, Y):
        preds = model.apply(params, X)
        return jnp.mean((preds - Y) ** 2)

    loss_and_grads = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(params, opt_state, X, Y):
        loss, grads = loss_and_grads(params, X, Y)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt_state, loss

    for epoch in range(epochs):
        rng, step_rng = jax.random.split(rng)
        indices = jax.random.permutation(step_rng, X.shape[0])
        X, Y = jnp.array(X)[indices], jnp.array(Y)[indices]

        for i in range(0, len(X), batch_size):
            X_batch, Y_batch = X[i:i+batch_size], Y[i:i+batch_size]
            params, opt_state, loss = train_step(params, opt_state, X_batch, Y_batch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return params

# ✅ **Load & Preprocess Data**
X, Y, sequences, mutation_ids = preprocess_data(FILE_PATH)

# ✅ **Train-test Split**
X_train, X_test, Y_train, Y_test, seq_train, seq_test, mut_train, mut_test = train_test_split(
    X, Y, sequences, mutation_ids, train_size=0.9, random_state=44
)

# Convert to JAX arrays
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
Y_train, Y_test = jnp.array(Y_train), jnp.array(Y_test)

# ✅ **Initialize and Train Model**
model = MLP(width=64, depth=3)
params = model.init(random.PRNGKey(0), jnp.ones((1, X_train.shape[1])))
params = train(model, params, (X_train, Y_train), epochs=100, lr=1e-3, batch_size=32)

# ✅ **Make Predictions**
Y_pred = model.apply(params, X_test)

# ✅ **Calculate R² Score**
r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(3)]
print(f"R² Scores: 0mM: {r2_scores[0]:.4f}, 1mM: {r2_scores[1]:.4f}, Fold Change: {r2_scores[2]:.4f}")

# ✅ **Save Predictions to CSV**
predictions_df = pd.DataFrame({
    "Mutation_ID": mut_test,
    "Modified_Sequence": seq_test,
    "True_0mM": Y_test[:, 0],
    "Predicted_0mM": Y_pred[:, 0],
    "True_1mM": Y_test[:, 1],
    "Predicted_1mM": Y_pred[:, 1],
    "True_Fold_Change": Y_test[:, 2],
    "Predicted_Fold_Change": Y_pred[:, 2]
})
predictions_file = os.path.join(SAVE_PATH, "mlp_predictions_test.csv")
predictions_df.to_csv(predictions_file, index=False)
print(f"✅ Predictions saved to: {predictions_file}")
