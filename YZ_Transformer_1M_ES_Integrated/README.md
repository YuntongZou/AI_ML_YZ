## **Gap-Aware Transformer for Predicting Fold Change from DNA Sequences**

### **Problem Definition**

This project develops a **Transformer-based deep learning model** that predicts **fold change values** based on DNA sequences with insertions/deletions (gaps). The gaps are inserted based on mutation IDs and are handled explicitly during tri-nucleotide encoding.

The model is built with **PyTorch** and is trained to regress the numeric **fold change** associated with each sequence using advanced Transformer encoder layers.

---

## **Input and Preprocessing**

The input is a CSV file with mutation-labeled DNA sequences and their fold change values under a specific condition.

### **Input Columns**

| Column Name      | Description                                        |
|------------------|----------------------------------------------------|
| **id**           | Mutation identifier (e.g., contains 'd#' for deletion sites) |
| **full_sequence**| Raw DNA sequence                                   |
| **fold_change**  | Numeric value representing biological fold change  |

### **Preprocessing Steps**

1. **Gap Insertion:**
   - For each sequence, a `-` (gap) is inserted at positions indicated by the mutation ID (e.g., `d25` inserts a gap at position 24).
   - This ensures all sequences are properly aligned for encoding.

2. **Tri-Nucleotide Encoding (Modified):**
   - Each 3-mer is encoded based on a fixed dictionary of 64 ATGC triplets.
   - Gaps (`---`) are treated specially and encoded as index `0`.

3. **Train-Test Split:**
   - The dataset is split into **90% training** and **10% testing**.

---

## **Model Overview**

The model follows the standard **Transformer Encoder** structure with:

- **Embedding Layer:**
  - Maps each tri-nucleotide token to a learned embedding.
  - Index `0` (gap) is treated as padding via `padding_idx=0`.

- **Positional Embedding:**
  - A trainable positional encoding vector is added to input embeddings.

- **Transformer Encoder:**
  - `2` layers, each with:
    - `4` self-attention heads
    - Feed-forward layers of width 256
    - GELU activation
    - Dropout regularization

- **Output Layer:**
  - Fully connected layer mapping to a single continuous output (fold change).

---

## **Training Procedure**

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Gradient Accumulation:** 4 steps to simulate larger batch size
- **Batch Size:** 16
- **Epochs:** 100
- **Model Checkpointing:** Best model saved based on R² score on the validation set

---

## **Evaluation & Output**

After training:

- The model is used to predict fold change on the held-out test set.
- Outputs are saved in a CSV file:
  - Mutation ID
  - Aligned Sequence (with inserted gaps)
  - True fold change
  - Predicted fold change

---

Yuntong Zou