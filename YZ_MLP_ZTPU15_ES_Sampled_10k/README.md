
## **MLP Model for Predicting Fold Change from DNA Sequences**

### **Problem Definition**
This project aims to train a **Multi-Layer Perceptron (MLP)** model to predict **fold change** values from DNA sequences. The model takes a **one-hot encoded representation** of DNA sequences and learns to map them to their corresponding **fold change values**, which are numerical representations of biological changes.

The model is implemented using **JAX** and **Flax**, enabling efficient training and inference on modern hardware.

---

## **Input and Preprocessing**
The input data consists of DNA sequences and their associated fold change values. The dataset is stored as a CSV file with the following columns:

| Column Name         | Description |
|--------------------|-------------|
| **Modified_Sequence** | The DNA sequence (string of A, T, G, C) |
| **new_fold_change** | The numerical fold change value for the sequence |

### **Preprocessing Steps**
1. **One-hot Encoding:**  
   - Each DNA sequence is converted into a **one-hot encoded vector** where each nucleotide (A, T, G, C) is represented by a binary vector.
   - The resulting input feature vector is flattened.
   
2. **Train-Test Split:**  
   - The dataset is split into **75% training** and **25% testing**.

---

## **Model Overview**
The model is a **Multi-Layer Perceptron (MLP)** with the following architecture:

- **Input Layer:** One-hot encoded sequence features  
- **Hidden Layers:**  
  - `depth` (default = 3) fully connected layers  
  - Each layer has `width` (default = 64) neurons  
  - **ReLU** activation function  
- **Output Layer:**  
  - Single neuron for predicting fold change  
  - No activation function (regression output)

The model is trained using the **Mean Squared Error (MSE) loss** and optimized using the **Adam optimizer**.

---

## **Training**
The model is trained using the following procedure:
1. **Initialize Parameters:**  
   - The parameters of the MLP are initialized using a random seed.
   
2. **Forward Pass:**  
   - The model processes the **one-hot encoded DNA sequences** and predicts fold change values.

3. **Loss Computation:**  
   - Mean Squared Error (MSE) is computed between predictions and true values.

4. **Backpropagation & Optimization:**  
   - The **Adam optimizer** updates the parameters based on the computed gradients.

5. **Epochs:**  
   - The model is trained for **100 epochs** with a batch size of **32**.

---

## **Expected Results**
The model aims to predict **fold change values** with high accuracy. The performance of the model is evaluated using:

- **R² Score (Coefficient of Determination):** Measures how well predictions match the actual values.
- **Scatter Plot:** Compares true vs. predicted fold change values.

---

## **Usage Instructions**
### **1. Install Dependencies**
Ensure you have JAX, Flax, NumPy, Pandas, and Matplotlib installed. You can install them using:
```bash
pip install -r requirements.txt

```

### **2. Run Training Script**
To train the model, execute:
```bash
python MLP_model.py
```

### **3. Save & Load Model**
The trained model is saved as a pickle file:
```python
save_model(params, "mlp_model.pkl")
loaded_params = load_model("mlp_model.pkl", model, example_input)
```

### **4. Predict on Test Data**
The model generates predictions on test data, which are saved in a CSV file:
```python
predictions_df.to_csv("mlp_test_predictions.csv", index=False)
```

---

## **Output Files**
After running the script, you will get:
- **`mlp_model.pkl`** → The trained model file.
- **`mlp_test_predictions.csv`** → The test set predictions.
- **Scatter plot** visualizing model performance.

---

## **Authors**
- **Yuntong Zou**  
