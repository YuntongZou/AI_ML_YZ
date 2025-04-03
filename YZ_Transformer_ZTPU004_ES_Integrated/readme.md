# **Integrated Transformer for Predicting ZTP Function Levels from DNA Sequences**

## **Problem Definition**
This project aims to train a deep learning model using a **Transformer-based architecture** to predict **ZTP function dose-response curves** from DNA sequences. The model processes **tri-nucleotide encoded** sequences and predicts **ZTP function levels across six concentration points (T1-T6)**.

The core of this model lies in its **deeply nested Transformer encoder blocks**, which allow the model to capture complex patterns and dependencies in DNA sequences beyond what simpler architectures can achieve.

---

## **Input and Preprocessing**
The input data is a CSV file with the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **Mutated_Sequence** | Mutated DNA sequence (A, T, G, C) |
| **T1 - T6**          | ZTP function levels at six concentration levels |

### **Preprocessing Steps**
1. **Tri-Nucleotide Encoding**  
   - Each sequence is broken into overlapping 3-mers.  
   - Each 3-mer is mapped to a unique integer index.  
   - The final encoded sequence is an integer vector of length *(L - 2)*, where *L* is the original sequence length.

2. **NaN Removal**  
   - Any rows containing NaNs in T1–T6 are removed.

3. **Train-Test Split**  
   - Dataset is split into **80% training** and **20% testing**.

---

## **Model Architecture**

The model is an **Integrated Transformer** consisting of the following components:

- **Embedding Layer**  
  Maps each tri-nucleotide index into a dense vector of dimension `embed_dim=64`.

- **Positional Embedding**  
  A learnable tensor is added to preserve sequence order.

- **Transformer Encoder Block (Nested)**  
  - Consists of **2 stacked encoder layers**, each with:
    - `nhead=4` multi-head self-attention.
    - Feedforward dimension = 256.
    - GELU activation and dropout.
  - The **nesting of these encoder layers** allows the model to capture deep contextual relationships between nucleotide triplets.

- **Mean Pooling Layer**  
  After the Transformer, a mean pooling operation summarizes the sequence.

- **Fully Connected Output Layer**  
  Maps pooled embeddings to a vector of size `6` (corresponding to the six predicted ZTP function levels).

---

## **Training Workflow**

1. **Model Instantiation**  
   - A single model is trained to predict all 6 ZTP function levels jointly.

2. **Loss Function**  
   - Uses **Mean Squared Error (MSE)** between predictions and true values.

3. **Optimization**  
   - Optimized using the **Adam optimizer** with learning rate `1e-3`.

4. **Evaluation**  
   - **Average R² score** across all 6 levels is computed for validation.

5. **Epochs**  
   - The model is trained for `100 epochs` with `batch size = 32`.

---

## **Output & Visualization**

After training, the following are saved:

| Output File | Description |
|-------------|-------------|
| `integrated_transformer_model.pt` | Trained PyTorch model file |
| `integrated_transformer_predictions.csv` | CSV of predicted vs. true values for test set |
| `transformer_r2_scatter_test.png` | Scatter plot of predicted vs. true values (T1–T6) |
| `transformer_curve_comparison_test.png` | Dose-response curve comparison for 5 random sequences |

---

## **Usage Instructions**

### **1. Install Required Packages**
```bash
pip install torch pandas numpy matplotlib scikit-learn
