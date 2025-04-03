# MLP for Predicting Fold Change from DNA Sequences

## **Problem Definition**
We train two separate **Multi-Layer Perceptron (MLP)** models to predict **0mM** and **1mM** binding levels from DNA sequences. These values are used to compute the **fold change** (1mM / 0mM), representing how much binding changes in response to Mg²⁺ concentration.

The models are implemented using **JAX** and **Flax**, providing efficient auto-differentiation and compilation for fast training and inference.

---

## **Input and Preprocessing**
The dataset is stored in a CSV file and contains DNA sequences with mutation annotations and measured binding data.

| Column Name        | Description |
|--------------------|-------------|
| **id**             | Mutation identifier (e.g., `d25` = delete position 25) |
| **full_sequence**  | Raw DNA sequence (string of A, T, G, C) |
| **0mM_fracBound**  | Binding level at 0mM Mg²⁺ |
| **1mM_fracBound**  | Binding level at 1mM Mg²⁺ |
| **fold_change**    | Optional original fold change column |

### **Preprocessing Steps**
1. **Gap Insertion Based on Mutation ID:**  
   - Gaps (`-`) are inserted into sequences at positions specified in the `id` column.
   - Ensures all sequences are aligned properly for consistent encoding.

2. **One-hot Encoding:**  
   - Each aligned sequence is one-hot encoded using the alphabet `["A", "T", "G", "C", "-"]`.
   - The resulting 2D matrix is flattened into a 1D feature vector.

3. **Train-Test Split:**  
   - The dataset is split into **90% training** and **10% testing**.

---

## **Model Overview**
Each model is a **Multi-Layer Perceptron (MLP)** defined using Flax:

- **Input Layer:**  
  - One-hot encoded DNA sequence (flattened).

- **Hidden Layers:**  
  - `depth = 3` fully connected layers  
  - `width = 64` neurons per layer  
  - Activation: **ReLU**

- **Output Layer:**  
  - A single neuron for regression (binding level prediction).

Two models are trained:
- One for **0mM_fracBound**
- One for **1mM_fracBound**

The final **fold change** is calculated as:
```
fold_change = predicted_1mM / (predicted_0mM + epsilon)
```

---

## **Training Process**
1. **Model Initialization:**  
   - Each MLP model is initialized with a random key and proper input shape.

2. **Forward Pass & Loss Computation:**  
   - Predicts binding levels using forward pass.
   - Loss is computed using **Mean Squared Error (MSE)**.

3. **Gradient Calculation:**  
   - Uses `jax.value_and_grad` for automatic differentiation.

4. **Optimization:**  
   - Optimized using **Adam** with learning rate `1e-3`.

5. **Training Loop:**  
   - Trained for **100 epochs** in batches of **32** samples.

6. **Saving Models:**  
   - Parameters for both models are serialized and saved using `flax.serialization`.

---

## **Evaluation & Prediction**
After training:

- Both models are reloaded from disk.
- Predictions are made on the test set for both 0mM and 1mM.
- Fold change is calculated for each test sample.
- R² scores are computed for:
  - 0mM prediction
  - 1mM prediction
  - Fold change prediction

Results are saved to a CSV file.

---

## **Output Files**

| File | Description |
|------|-------------|
| **`params_model_0mM.pkl`** | Trained parameters for the 0mM model |
| **`params_model_1mM.pkl`** | Trained parameters for the 1mM model |
| **`mlp_predictions_from_loaded_models.csv`** | Test predictions and true values for 0mM, 1mM, and fold change |

---


## **Expected Results**
The trained models aim to predict:

- **0mM and 1mM binding levels** with high accuracy
- **Derived fold change** close to experimental measurements

The output CSV enables easy inspection and further visualization of model performance.

---

## **Author**
**Yuntong Zou**
