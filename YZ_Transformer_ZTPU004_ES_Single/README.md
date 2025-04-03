
# **Transformer Model for Predicting Fluorescence Dose Curves from DNA Sequences**

## **Problem Definition**
This project utilizes a **Transformer-based deep learning model** to predict **fluorescence dose-response curves** from DNA sequences. The model processes **tri-nucleotide encoded representations** of DNA sequences and predicts **fluorescence intensity values** across six concentration levels.

The model is implemented using **PyTorch**, leveraging self-attention mechanisms to improve sequence-based regression performance.

---

## **Input and Preprocessing**
The input dataset is stored as a CSV file and contains the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **Mutated_Sequence** | DNA sequence (string of A, T, G, C) |
| **T1 - T6** | Fluorescence intensity values at six different concentrations |

### **Preprocessing Steps**
1. **Tri-Nucleotide Encoding:**  
   - Each DNA sequence is converted into a **tri-nucleotide encoded vector**, representing overlapping 3-mers as unique numerical indices.
   - The resulting feature vector is used as input to the Transformer model.

2. **Handling Missing Data:**  
   - Any row containing **NaN values** across fluorescence intensity columns is removed.

3. **Train-Test Split:**  
   - The dataset is divided into **80% training** and **20% testing**.

---

## **Model Overview**
The model is a **Transformer Encoder Network** with the following architecture:

- **Input Layer:**  
  - DNA sequences are mapped to numerical indices based on their tri-nucleotide representation.  
  - Each index is embedded using a learnable **embedding layer**.  
  - A **positional encoding layer** is added to retain sequence order information.  

- **Transformer Encoder:**  
  - `num_layers=2` self-attention blocks.  
  - `num_heads=4` attention heads.  
  - `feedforward_dim=256` with dropout regularization (`dropout=0.1`).  

- **Output Layer:**  
  - A **fully connected layer** that outputs a single fluorescence intensity prediction.  

The model is trained using the **Mean Squared Error (MSE) loss** function and optimized using **Adam optimizer**.

---

## **Training Process**
The model is trained separately for each fluorescence intensity level (`T1 - T6`). The steps are:

1. **Initialize Model & Parameters:**  
   - A new Transformer model instance is trained per fluorescence intensity value.

2. **Forward Pass:**  
   - The model processes **tri-nucleotide encoded DNA sequences** to predict fluorescence values.

3. **Loss Computation:**  
   - The **Mean Squared Error (MSE)** between predictions and true values is calculated.

4. **Backpropagation & Optimization:**  
   - The **Adam optimizer** updates model weights based on computed gradients.

5. **Epochs & Evaluation:**  
   - The model is trained for **100 epochs** with a batch size of **32**.
   - The best model per intensity level is saved based on **R² score**.

---

## **Expected Results**
The model aims to achieve high accuracy in predicting fluorescence dose curves. The performance is evaluated using:

- **R² Score (Coefficient of Determination):**  
  Measures the goodness of fit between predictions and actual values.

- **Scatter Plot:**  
  Compares **true vs. predicted fluorescence values** for each concentration level.

- **Dose-Response Curve Comparison:**  
  - Selects **five random test sequences** and plots their **true vs. predicted fluorescence response** across six concentrations.

---


To load a trained model for inference:
```python
model.load_state_dict(torch.load("transformer_model_T1.pt", map_location=device))
```

### **4. Generate Predictions & Save Results**
The final test set predictions are saved in:
```bash
/projects/p32603/1_FL_Curve/transformer_test_predictions.csv
```

---

## **Output Files**
After running the script, the following files will be generated:

| Output File | Description |
|------------|-------------|
| **`transformer_model_T{i}.pt`** | Trained Transformer model for each fluorescence intensity level (T1-T6) |
| **`transformer_test_predictions.csv`** | Predicted fluorescence intensities for test set |
| **`transformer_r2_scatter_test.png`** | Scatter plot comparing true vs. predicted test values |
| **`transformer_curve_comparison_test.png`** | Plot of predicted vs. actual dose-response curves |

---

## **Visualizations**
1. **R² Score Scatter Plot**  
   - Compares **true vs. predicted fluorescence intensities**.  
   - Helps evaluate model performance across different concentration levels.  

2. **Dose-Response Curve Comparison**  
   - Displays **true vs. predicted curves** for randomly selected test sequences.  
   - Demonstrates how well the model captures fluorescence dynamics.

---

- **Yuntong Zou**  

