# **Integrated Transformer for Predicting ZTP Function Curves from DNA Sequences**

## **Problem Definition**
This project aims to train a deep learning model to predict **ZTP function dose-response curves** from mutated DNA sequences. The model receives **tri-nucleotide encoded DNA sequences** and predicts **ZTP function levels at six concentrations (T1–T6)**.

The model leverages a **stacked Transformer Encoder** architecture to effectively learn complex sequence-level representations, capturing both local motifs and global dependencies in DNA.

---

## **Input and Preprocessing**

### **Input Format**
The input is a CSV file with the following structure:

| Column Name         | Description |
|--------------------|-------------|
| **Mutated_Sequence** | DNA sequence (A, T, G, C) with optional mutations |
| **T1 - T6**          | ZTP function levels under 6 concentration conditions |

### **Preprocessing Steps**
1. **Tri-Nucleotide Encoding**
   - Sequences are encoded into overlapping 3-mers (e.g., `ATG`, `TGC`, `GCA`...).
   - Each 3-mer is mapped to a unique index, resulting in a sequence of integers.

2. **Missing Value Removal**
   - Rows with any `NaN` in the T1–T6 columns are dropped.

3. **Train-Test Split**
   - The data is split into **80% training** and **20% testing**.

---

## **Model Architecture**

The architecture is an **Integrated Transformer Encoder**, composed of the following:

### 🔷 **Input Layer**
- **Embedding Layer**:  
  - Converts each tri-nucleotide index into a 64-dimensional learnable vector.  
  - `nn.Embedding(num_embeddings=64, embedding_dim=64)`

- **Positional Encoding**:  
  - A learnable tensor of shape `(1, sequence_length, 64)` is added to the embedding to retain positional order.  
  - `nn.Parameter(torch.randn(1, input_dim, embed_dim))`

---

### 🔷 **Nested Transformer Block**
This is the core component of the model. It includes **two stacked TransformerEncoder layers**:

Each Transformer layer contains:
- **Multi-Head Self Attention**:
  - `nhead=4` attention heads.
  - Enables the model to attend to different positions in the sequence simultaneously.

- **Feedforward Network**:
  - A fully connected sub-network with hidden size 256.
  - Two linear layers and a GELU activation.
  - Dropout rate: `0.1` for regularization.

- **LayerNorm and Residual Connections**:
  - Applied before and after attention and feedforward blocks for training stability.

### Code Representation:
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=256, dropout=0.1,
    activation="gelu", batch_first=True
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
