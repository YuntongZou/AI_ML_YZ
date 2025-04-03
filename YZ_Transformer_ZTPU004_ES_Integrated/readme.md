## **Integrated Transformer for Predicting Dose-Response Curves from Mutated DNA Sequences**

### **Problem Definition**

This project implements an **Integrated Transformer model** to predict **dose-response curves** (i.e., fold change values across 6 concentrations) from mutated DNA sequences. Each DNA sequence is processed using **tri-nucleotide encoding**, which captures local sequence context. The Transformer learns to predict all 6 response values simultaneously.

---

## **Input and Preprocessing**

The input is a `.csv` file containing mutated DNA sequences and 6 associated fold change measurements at different concentrations.

### **Input Columns**

| Column Name         | Description |
|---------------------|-------------|
| **Mutated_Sequence** | The DNA sequence after mutation |
| **T1-T6** (ZMP levels) | function values at 6 concentrations (stored in columns 2 to 7) |

### **Preprocessing Steps**

1. **Tri-nucleotide Encoding:**
   - Each sequence is converted into a sequence of tri-nucleotide indices using a vocabulary of all 64 possible 3-mers.
   - Encoded sequences are stored as integer arrays for model input.

2. **Train-Test Split:**
   - The dataset is randomly split into **80% training** and **20% testing**.

---

## **Model Overview**

The model is a Transformer-based architecture with the following components:

- **Input Layer:**
  - Tri-nucleotide integer indices
  - Embedding layer maps each index to a feature vector
  - Positional embedding is added to preserve sequence order

- **Transformer Encoder:**
  - `num_layers = 2` transformer blocks
  - Each block uses `num_heads = 4` attention heads
  - Feedforward layer with 256 hidden units

- **Output Layer:**
  - Fully connected layer mapping encoded features to 6 outputs
  - Predicts fold change at 6 concentrations

---

## **Training**

The model is trained using the following pipeline:

1. **Loss Function:**
   - **Mean Squared Error (MSE)** between predicted and true values for all 6 outputs

2. **Optimizer:**
   - **Adam optimizer** with a learning rate of `1e-3`

3. **Epochs and Batch Size:**
   - Trained for **100 epochs**
   - Batch size: **32**

4. **Validation and Saving:**
   - R² score is calculated for each output
   - The **average R²** across 6 outputs is tracked
   - The best model is saved when average R² improves

---

## **Expected Results**

The model is expected to learn the nonlinear mapping between mutated sequences and their dose-response behavior.

- **R² Scores** for each of the 6 outputs are reported
- **Scatter plot** compares predicted vs. true values
- **Line plots** show predicted vs. true dose-response curves for 5 sample sequences

---

Yuntong Zou