
# **HMM-Transformer Model for Predicting Fluorescence Dose Curves from DNA Mutations**

## **Problem Definition**
This project integrates **Hidden Markov Models (HMMs)** with **Transformer-based deep learning** to predict **fluorescence dose-response curves** from DNA mutations. The model learns to map **nucleotide sequence mutations** to **fluorescence intensities** across **six concentration levels** using a hybrid **HMM + Transformer** architecture.

The **HMM** captures underlying latent **states** in fluorescence response, while the **Transformer** learns complex sequence dependencies, improving interpretability and predictive power.

The model is implemented in **PyTorch** and **hmmlearn**.

---

## **Input and Preprocessing**
The input dataset is stored as a CSV file and contains the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **mutation_site** | DNA mutation sequence (string of A, T, G, C) |
| **T1 - T6** | Fluorescence intensity values at six different concentrations |

### **Preprocessing Steps**
1. **One-hot Encoding for Mutations:**  
   - Each DNA sequence is **one-hot encoded** using a mapping:  
     - A → `[1,0,0,0]`  
     - T → `[0,1,0,0]`  
     - G → `[0,0,1,0]`  
     - C → `[0,0,0,1]`  
   - This results in a **(N, sequence_length, 4)** feature representation.

2. **HMM State Prediction:**  
   - A **7-state Gaussian Hidden Markov Model (HMM)** is trained on the fluorescence intensity values.
   - The **HMM assigns states** to each fluorescence measurement across six time points.

3. **Train-Test Split:**  
   - The dataset is divided into **80% training** and **20% testing**, ensuring all fluorescence levels (T1-T6) are split consistently.

---

## **Model Overview**
The **HMM-Transformer Model** combines:
1. **HMM Component:**  
   - Learns a **hidden state representation** from fluorescence values.  
   - Encodes fluorescence response as discrete **latent states**.

2. **Transformer Component:**  
   - **Embeds both mutation sequences & HMM states**.  
   - **Self-attention** mechanism models complex dependencies.  
   - Outputs **six fluorescence intensity values simultaneously**.

### **Architecture**
- **Mutation Embedding:**  
  - A **fully connected layer** processes one-hot encoded mutation sequences.
  
- **HMM State Embedding:**  
  - A **learnable embedding layer** represents discrete **HMM states**.

- **Attention Mechanism:**  
  - **Queries (Q):** Mutation sequence embedding.  
  - **Keys (K) & Values (V):** HMM state embedding.  
  - Computes **self-attention over nucleotide mutations and HMM states**.

- **Transformer Encoder:**  
  - `num_layers=2` self-attention layers.  
  - `num_heads=4` attention heads.  
  - Outputs a **feature vector** for regression.

- **Fully Connected Layer:**  
  - Produces **six fluorescence intensity values (T1-T6)**.

The model is trained using **Mean Squared Error (MSE) loss** and optimized using **Adam optimizer**.

---

## **Training Process**
The model is trained using the following procedure:

1. **Train Hidden Markov Model (HMM):**  
   - The HMM is **fitted on fluorescence values**, learning **hidden states**.

2. **Train Transformer Model:**  
   - Inputs: **One-hot encoded mutations + HMM latent states**.  
   - Outputs: **Fluorescence intensities (T1-T6)**.  
   - **Loss Function:** Mean Squared Error (MSE).  
   - **Optimizer:** Adam.  
   - **Epochs:** 50.

3. **Evaluation on Test Data:**  
   - Compute **MSE & R² Score**.
   - Compare **true vs. predicted fluorescence intensity**.

---

## **Expected Results**
The model aims to achieve **high accuracy** in predicting fluorescence dose-response curves.

- **Average R² Score:**  
  Measures how well the predicted fluorescence values match the actual values.

- **Prediction Error Distribution:**  
  - Histogram of **prediction errors** to analyze model performance.

- **Dose-Response Curve Comparison:**  
  - Selects **five random test sequences** and plots their **true vs. predicted fluorescence response**.

- **Fold Change Analysis:**  
  - Computes **fold change (T6 / T1)** for predicted & true values.
  - **R² Score for Fold Change** measures the ability to **capture relative changes**.

---


### **3. Generate Predictions & Save Results**
The final test set predictions are saved in:
```bash
/projects/p32603/HMM_model/HMM+transformer/reverse_HMM/test_predictions.csv
```

---

## **Output Files**
After running the script, the following files will be generated:

| Output File | Description |
|------------|-------------|
| **`hmm_transformer_model.pt`** | Trained HMM-Transformer model for predicting fluorescence values |
| **`test_predictions.csv`** | Predicted fluorescence intensities for test set |
| **`predictions_2.png`** | Predicted vs. actual fluorescence curves |
| **`error_distribution_2.png`** | Histogram of prediction errors |
| **`correlation_scatter_2.png`** | True vs. predicted fluorescence scatter plot |
| **`fold_change_r2.png`** | R² Score for fold change (T6 / T1) |

---

## **Example Visualizations**
1. **R² Score Scatter Plot**  
   - Compares **true vs. predicted fluorescence intensities**.  
   - Helps evaluate model performance across different concentration levels.  

2. **Dose-Response Curve Comparison**  
   - Displays **true vs. predicted curves** for randomly selected test sequences.  
   - Demonstrates how well the model captures fluorescence dynamics.

3. **Fold Change Analysis**  
   - Computes and compares **true vs. predicted fold change** values.  
   - High **R² Score** indicates **accurate ratio predictions**.

---

## **Key Improvements Over Previous Models**
✅ **HMM-Enhanced Predictions:** Learns **latent states** from fluorescence curves, improving sequence interpretation.  
✅ **Attention-Based Fusion:** Captures **sequence dependencies & HMM states** jointly.  
✅ **Multi-Output Learning:** Predicts **all six fluorescence values simultaneously**, making inference efficient.  
✅ **Fold Change R² Analysis:** Evaluates model ability to **capture relative intensity changes**.  

- **Yuntong Zou**  

