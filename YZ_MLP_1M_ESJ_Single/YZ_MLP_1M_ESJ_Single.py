import os
import pickle
import numpy as np
import pandas as pd
import re
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from jax import random
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ Set Paths
FILE_PATH = "/projects/p32363/AI_ML_Data/ZTP_1M_GreB_LE_binary_curve_combined_with_sequences.csv"
SAVE_PATH = "/save/to/your/path"
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ One-hot encoding
def one_hot_encode(sequence, alphabet="ATGC-"):
    mapping = {char: i for i, char in enumerate(alphabet)}
    one_hot = np.zeros((len(sequence), len(alphabet)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot.flatten()

# ✅ Insert gaps based on mutation ID
def insert_gaps(sequence, mutation_id):
    positions_to_delete = sorted([int(pos[1:]) for pos in re.findall(r'd\d+', mutation_id)])  
    sequence = list(sequence)  
    for pos in positions_to_delete:
        if pos - 1 < len(sequence):
            sequence.insert(pos - 1, "-")  
    return "".join(sequence)

# ✅ Preprocess data
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    required_cols = ["id", "full_sequence", "0mM_fracBound", "1mM_fracBound", "fold_change"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset!")
    mutation_ids = df["id"].values
    sequences = df["full_sequence"].values
    y_bound = df["0mM_fracBound"].values  
    y_unbound = df["1mM_fracBound"].values  
    valid_indices = ~np.isnan(y_bound) & ~np.isnan(y_unbound)
    mutation_ids, sequences, y_bound, y_unbound = (
        mutation_ids[valid_indices], sequences[valid_indices], 
        y_bound[valid_indices], y_unbound[valid_indices]
    )
    aligned_sequences = [insert_gaps(seq, mut_id) for seq, mut_id in zip(sequences, mutation_ids)]
    X = np.array([one_hot_encode(seq) for seq in aligned_sequences])
    return X, y_bound, y_unbound, aligned_sequences, mutation_ids

# ✅ Define MLP
class MLP(nn.Module):
    width: int
    depth: int
    def setup(self):
        self.layers = [nn.Dense(self.width) for _ in range(self.depth)]
        self.output_layer = nn.Dense(1)
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return self.output_layer(x).squeeze()

# ✅ Training function
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

# ✅ Load and preprocess data
X, y_bound, y_unbound, sequences, mutation_ids = preprocess_data(FILE_PATH)
X_train, X_test, yb_train, yb_test, yu_train, yu_test, seq_train, seq_test, mut_train, mut_test = train_test_split(
    X, y_bound, y_unbound, sequences, mutation_ids, train_size=0.9, random_state=44
)
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
yb_train, yb_test = jnp.array(yb_train), jnp.array(yb_test)
yu_train, yu_test = jnp.array(yu_train), jnp.array(yu_test)

# ✅ Train two models
model_0mM = MLP(width=64, depth=3)
params_0mM = model_0mM.init(random.PRNGKey(0), jnp.ones((1, X_train.shape[1])))
params_0mM = train(model_0mM, params_0mM, (X_train, yb_train), epochs=100, lr=1e-3)

model_1mM = MLP(width=64, depth=3)
params_1mM = model_1mM.init(random.PRNGKey(1), jnp.ones((1, X_train.shape[1])))
params_1mM = train(model_1mM, params_1mM, (X_train, yu_train), epochs=100, lr=1e-3)

# ✅ Save parameters
with open(os.path.join(SAVE_PATH, "params_model_0mM.pkl"), "wb") as f:
    pickle.dump(serialization.to_state_dict(params_0mM), f)
with open(os.path.join(SAVE_PATH, "params_model_1mM.pkl"), "wb") as f:
    pickle.dump(serialization.to_state_dict(params_1mM), f)
print("✅ Model parameters saved.")

# ✅ Load models for prediction
with open(os.path.join(SAVE_PATH, "params_model_0mM.pkl"), "rb") as f:
    state_dict_0mM = pickle.load(f)
params_0mM = serialization.from_state_dict(model_0mM.init(jax.random.PRNGKey(0), X_test[:1]), state_dict_0mM)

with open(os.path.join(SAVE_PATH, "params_model_1mM.pkl"), "rb") as f:
    state_dict_1mM = pickle.load(f)
params_1mM = serialization.from_state_dict(model_1mM.init(jax.random.PRNGKey(1), X_test[:1]), state_dict_1mM)

# ✅ Predict
yb_pred = model_0mM.apply(params_0mM, X_test)
yu_pred = model_1mM.apply(params_1mM, X_test)
epsilon = 1e-6
fc_pred = yu_pred / (yb_pred + epsilon)
fc_true = yu_test / (yb_test + epsilon)

# ✅ R² Scores
r2_0mM = r2_score(yb_test, yb_pred)
r2_1mM = r2_score(yu_test, yu_pred)
r2_fc = r2_score(fc_true, fc_pred)
print(f"R² Scores:\n  0mM: {r2_0mM:.4f}\n  1mM: {r2_1mM:.4f}\n  Fold Change: {r2_fc:.4f}")

# ✅ Save predictions to CSV
predictions_df = pd.DataFrame({
    "Mutation_ID": mut_test,
    "Modified_Sequence": seq_test,
    "True_0mM": yb_test,
    "Predicted_0mM": yb_pred,
    "True_1mM": yu_test,
    "Predicted_1mM": yu_pred,
    "True_Fold_Change": fc_true,
    "Predicted_Fold_Change": fc_pred
})
csv_path = os.path.join(SAVE_PATH, "mlp_predictions_from_loaded_models.csv")
predictions_df.to_csv(csv_path, index=False)
print(f"✅ Predictions saved to: {csv_path}")
