
# **MLP Model for Predicting Fluorescence Dose Curves from DNA Sequences**

## **Problem Definition**
We train a **Multi-Layer Perceptron (MLP)** model to predict **dose-response curves** from DNA sequences. The model processes a **one-hot encoded representation** of DNA sequences and predicts **function levels** across six different ZMP concentration levels.

The model is implemented using **PyTorch**.

---

## **Input and Preprocessing**
The input dataset is stored as a CSV file and contains the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **Mutated_Sequence** | DNA sequence (string of A, T, G, C) |
| **T1 - T6** | Functionlevel of ZTP at six different ZMP concentrations |

### **Preprocessing Steps**
1. **One-hot Encoding:**  
   - Each DNA sequence is converted into a **one-hot encoded vector**, representing nucleotides (A, T, G, C) as a binary feature space.
   - The resulting feature vector is flattened for model input.

2. **Handling Missing Data:**  
   - Any row containing **NaN values** across the fluorescence intensity columns is removed.

3. **Train-Test Split:**  
   - The dataset is divided into **80% training** and **20% testing**.

---

## **Model Overview**
The model is a **Multi-Layer Perceptron (MLP)** with the following architecture:

- **Input Layer:** One-hot encoded DNA sequence features.  
- **Hidden Layers:**  
  - `depth=3` fully connected layers  
  - `hidden_dim=128` neurons per layer  
  - **ReLU** activation function  
  - **Layer Normalization** for stable training  
  - **Dropout (0.2)** for regularization  
- **Output Layer:**  
  - A single neuron per fluorescence intensity prediction.  
  - No activation function (regression output).

The model is trained using the **Mean Squared Error (MSE) loss** function and optimized using **Adam optimizer**.

---

## **Training Process**
The model is trained for **each fluorescence intensity level separately (T1-T6)**. The steps are:

1. **Initialize Model & Parameters:**  
   - Each MLP instance is trained independently for predicting one fluorescence value at a time.

2. **Forward Pass:**  
   - The model processes **one-hot encoded DNA sequences** to predict fluorescence values.

3. **Loss Computation:**  
   - The Mean Squared Error (MSE) between predicted and true values is calculated.

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
  Compares **true vs. predicted values** for each fluorescence intensity.

- **Dose-Response Curve Comparison:**  
  - Selects **five random test sequences** and plots their **true vs. predicted fluorescence response** across six concentrations.

## **Output Files**
After running the script, the following files will be generated:

| Output File | Description |
|------------|-------------|
| **`mlp_model_T{i}.pt`** | Trained MLP model for each fluorescence intensity level (T1-T6) |
| **`mlp_test_predictions.csv`** | Predicted fluorescence intensities for test set |
| **`mlp_r2_scatter_test.png`** | Scatter plot comparing true vs. predicted test values |
| **`mlp_curve_comparison_test.png`** | Plot of predicted vs. actual dose-response curves |

---

## **Example Visualizations**
1. **R² Score Scatter Plot**  
   - Compares **true vs. predicted fluorescence intensities**.  
   - Helps evaluate model performance across different concentration levels.  

2. **Dose-Response Curve Comparison**  
   - Displays **true vs. predicted curves** for randomly selected test sequences.  
   - Demonstrates how well the model captures fluorescence dynamics.
   Author: Yuntong Zou
