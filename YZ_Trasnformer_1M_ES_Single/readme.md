## **Transformer Model for Predicting Bound and Unbound Fractions from DNA Sequences**

### **Problem Definition**
This project focuses on training a **Transformer-based deep learning model** to predict **0mM and 1mM fracBound values** from DNA sequences. These values represent biological binding behaviors under different molecular conditions. The input DNA sequences may contain mutations that introduce gaps, requiring careful preprocessing.

---

## **Input and Preprocessing**

The dataset is a CSV file containing DNA sequences with mutation IDs and their respective bound/unbound values. Columns include:

| Column Name        | Description                              |
|--------------------|------------------------------------------|
| **id**             | Mutation identifier string                |
| **full_sequence**  | Original DNA sequence                    |
| **0mM_fracBound**  | FracBound value with 0 mM input condition|
| **1mM_fracBound**  | FracBound value with 1 mM input condition|

### **Preprocessing Steps**
1. **Gap Insertion:**
   - Mutation identifiers specify deletion positions (e.g. `d32`, `d57`). Gaps (`-`) are inserted at these positions.
   - A binary **mask vector** is generated to mark real vs. gap positions.

2. **Tri-Nucleotide Encoding:**
   - Every sequence is encoded using overlapping tri-nucleotide triplets (e.g., `ATG`, `TGC`, etc.)
   - Each triplet is mapped to an integer.
   - Triplets containing any gap are masked out.

3. **Train-Test Split:**
   - Data is randomly split: **90% training**, **10% testing**.

---

## **Model Overview**

A **Transformer encoder model** is used with the following architecture:

- **Input Embedding Layer:**
  - Maps each tri-nucleotide integer to a learned vector (embedding).
  - Includes a learned **positional embedding** to retain positional information.

- **Transformer Encoder:**
  - 2 layers with 4 attention heads each
  - Uses **GELU activation** and dropout for regularization

- **Masking Logic:**
  - Attention is masked to ignore triplets derived from gaps.
  - The output is mean-pooled over valid (non-gap) positions.

- **Final Layer:**
  - A linear regression layer produces a single value prediction.

Two models are trained independently:
- One for **0mM_fracBound**
- One for **1mM_fracBound**

---

## **Training Procedure**
1. **Input Preparation:**
   - Sequences are one-hot/triplet encoded and masked.
   - Converted into PyTorch `TensorDataset` objects.

2. **Optimization:**
   - Loss: **Mean Squared Error (MSE)**
   - Optimizer: **Adam**
   - Learning rate scheduling: **ReduceLROnPlateau**

3. **Validation & Early Stopping:**
   - Performance is evaluated each epoch on the validation set using **R² score**.
   - Best-performing model (based on R²) is saved.

4. **Epochs:**
   - Model is trained for **100 epochs** with a batch size of **32**.

---

## **Expected Results**
The trained models aim to accurately predict **fracBound** values under 0mM and 1mM conditions.

Evaluation metrics include:
- **R² Score**: Measures prediction accuracy
- **Loss Curves**: Can be visualized optionally

## **Output Files**
After training, the script will generate:
- **`transformer_model_bound.pt`** → Model for 0mM fracBound prediction
- **`transformer_model_unbound.pt`** → Model for 1mM fracBound prediction
- Training loss and R² will be printed per epoch

---

## **Authors**
- **Yuntong Zou**  
