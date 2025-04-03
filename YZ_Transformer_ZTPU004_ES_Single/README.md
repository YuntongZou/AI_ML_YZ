# **Transformer Model for Predicting ZTP Function Curves from DNA Sequences**

## **Problem Definition**
This project utilizes a **Transformer-based deep learning model** to predict **ZTP function dose-response curves** from DNA sequences. The model processes **tri-nucleotide encoded representations** of DNA sequences and predicts **ZTP function levels** across six concentration levels.

The model is implemented using **PyTorch**, leveraging self-attention mechanisms to improve sequence-based regression performance.

---

## **Input and Preprocessing**
The input dataset is stored as a CSV file and contains the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **Mutated_Sequence** | DNA sequence (string of A, T, G, C) |
| **T1 - T6** | ZTP function levels at six different concentrations |

### **Preprocessing Steps**
1. **Tri-Nucleotide Encoding:**  
   - Each DNA sequence is converted into a **tri-nucleotide encoded vector**, representing overlapping 3-mers as unique numerical indices.
   - The resulting feature vector is used as input to the Transformer model.

2. **Handling Missing Data:**  
   - Any row containing **NaN values** across ZTP function columns is removed.

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
  - A **fully connected layer** that outputs a single ZTP function level prediction.  

The model is trained using the **Mean Squared Error (MSE) loss** function and optimized using **Adam optimizer**.

---

## **Training Process**
The model is trained separately for each ZTP function level (`T1 - T6`). The steps are:

1. **Initialize Model & Parameters:**  
   - A new Transformer model instance is trained per ZTP function level.

2. **Forward Pass:**  
   - The model processes **tri-nucleotide encoded DNA sequences** to predict ZTP function levels.

3. **Loss Computation:**  
   - The **Mean Squared Error (MSE)** between predictions and true values is calculated.

4. **Backpropagation & Optimization:**  
   - The **Adam optimizer** updates model weights based on computed gradients.

5. **Epochs & Evaluation:**  
   - The model is trained for **100 epochs** with a batch size of **32**.
   - The best model per concentration level is saved based on **R² score**.

---

## **Expected Results**
The model aims to achieve high accuracy in predicting ZTP function dose curves. The performance is evaluated using:

- **R² Score (Coefficient of Determination):**  
  Measures the goodness of fit between predictions and actual values.

- **Scatter Plot:**  
  Compares **true vs. predicted ZTP function levels** for each concentration level.

- **Dose-Response Curve Comparison:**  
  - Selects **five random test sequences** and plots their **true vs. predicted ZTP function response** across six concentrations.

---

## **Usage Instructions**

### **1. Install Dependencies**
Ensure you have the required Python libraries installed:
```bash
pip install -r requirements.txt
