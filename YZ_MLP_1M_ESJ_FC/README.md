# **MLP Model for Joint Prediction of Binding and Fold Change from DNA Sequences**

## **Problem Definition**
This project implements a **Multi-Layer Perceptron (MLP)** model to jointly predict **0mM binding**, **1mM binding**, and **fold change** from DNA sequences. The model processes a **one-hot encoded representation** of gap-aligned DNA sequences and outputs all three biological quantities simultaneously.

The model is implemented using **JAX** and **Flax**, allowing for high-performance automatic differentiation and parallelized model training.

---

## **Input and Preprocessing**
The input dataset is a CSV file containing mutation annotations and measured binding values:

| Column Name         | Description |
|---------------------|-------------|
| **id**              | Mutation ID specifying where deletions (`d25`) occur |
| **full_sequence**   | Full DNA sequence (A, T, G, C) |
| **0mM_fracBound**   | Binding level at 0 mM Mg²⁺ |
| **1mM_fracBound**   | Binding level at 1 mM Mg²⁺ |
| **fold_change**     | Ratio of 1mM / 0mM binding (precomputed) |

### **Preprocessing Steps**
1. **Gap Insertion:**  
   - For each sequence, positions indicated by `id` (e.g., `d25`) are replaced with gap symbols (`-`) to align all sequences.

2. **One-hot Encoding:**  
   - Each aligned sequence is converted into a one-hot encoded array using the alphabet `["A", "T", "G", "C", "-"]`.  
   - The 2D encoding is flattened into a single 1D feature vector.

3. **Target Preparation:**  
   - The model targets are stacked together:  
     `[0mM_binding, 1mM_binding, fold_change]`

4. **Train-Test Split:**  
   - The dataset is split into **90% training** and **10% testing**.

---

## **Model Overview**
The model is a simple yet expressive **Multi-Layer Perceptron (MLP)** defined in **Flax**, with the following structure:

- **Input:**  
  - Flattened one-hot encoded sequence vector

- **Hidden Layers:**  
  - `depth = 3` fully connected layers  
  - `width = 64` neurons per layer  
  - **ReLU** activation applied after each layer

- **Output Layer:**  
  - 3 output neurons (for 0mM, 1mM, and fold change)  
  - No activation (regression task)

The model uses **Mean Squared Error (MSE)** loss and is optimized using the **Adam optimizer**.

---

## **Training Procedure**
The model is trained for **100 epochs** with a batch size of **32**, using the following pipeline:

1. **Model Initialization:**  
   - Parameters are initialized using JAX's `random.PRNGKey`.

2. **Training Loop:**  
   - The training loop performs batch-wise updates of the parameters using gradients computed by `jax.value_and_grad`.

3. **Optimization:**  
   - Parameter updates are applied using `optax.adam`.

4. **Evaluation:**  
   - Every 10 epochs, the training loss is printed.  
   - After training, predictions are generated for the test set.

---

## **Expected Results**
The model is expected to output accurate predictions for:

- **0mM binding fraction**
- **1mM binding fraction**
- **Fold change (1mM / 0mM)**

The final **R² score** is computed for each of the three outputs to assess model performance.

---



## **Output Files**

| File Name                      | Description |
|--------------------------------|-------------|
| **`mlp_predictions_test.csv`** | Predicted and true values for 0mM, 1mM, and fold change on test data |

Each row includes:
- Mutation ID  
- Aligned (gapped) DNA sequence  
- True and predicted values for all three targets

---

## **Evaluation Metrics**
The model's performance is evaluated using **R² (coefficient of determination)** for each output:
- `R²_0mM`, `R²_1mM`, and `R²_fold_change`

These scores are printed at the end of training and used to assess model quality.

---

## **Author**
**Yuntong Zou**
