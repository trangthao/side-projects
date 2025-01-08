# **Attention-based NLP Classification**

This project implements a text classification model using attention mechanisms. I used GloVe embeddings for vectorization and processed consumer complaints to classify them into various financial product categories.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Requirements](#requirements)

---

## **Overview**

This project focuses on:
1/ Preprocessing consumer complaint narratives for classification.
2/ Using pre-trained GloVe embeddings to encode text data.
3/ Training an attention-based neural network for multi-class classification.
4/ Optimizing and saving the trained model for future inference.

---

## **Dataset**

- **Source:** The dataset consists of consumer complaints, including:
  - `Consumer complaint narrative`: Text describing the complaint.
  - `Product`: Target variable with financial product categories.

- **Key Statistics:**
  - Classes: Multiple financial product categories (e.g., loans, savings accounts, credit reports).
  - Features: Textual complaint data.

---

## **Model Architecture**

- **Embeddings:** Pre-trained GloVe vectors (50 dimensions) are used for encoding text.
- **Attention Mechanism:** Allows the model to focus on key parts of the input sequence.
- **Layers:**
  - Embedding Layer (using GloVe vectors)
  - Bidirectional LSTM for sequence encoding
  - Attention Layer for focusing on important tokens
  - Fully Connected Layer for classification

---

## **Requirements**

- Python 3.8 or higher
- Libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `tqdm`
