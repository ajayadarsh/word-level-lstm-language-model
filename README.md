# 🕵️ Word-Level Language Model
### Agatha Christie Style Text Generation using TensorFlow LSTM

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)](https://keras.io)
[![NLP](https://img.shields.io/badge/NLP-Language%20Model-green)](#)

---

## Overview

This project implements a **word-level language model** trained on *Poirot Investigates* by Agatha Christie using **TensorFlow/Keras**.

The model learns sequential word patterns from approximately **53,000 words** of detective fiction and generates new text in a style inspired by Agatha Christie.

Given a short prompt such as:

```text
Poirot looked at me and said
```

the model predicts the next sequence of words to generate coherent text.

---

## Final Results

| Metric | Result |
|---|---|
| Training Tokens | ~53,000 |
| Vocabulary Size | ~10k–12k |
| Model Type | Word-Level LSTM |
| Framework | TensorFlow/Keras |
| Sequence Length | 30 |
| Best Validation Loss | ~5.27 |
| Text Generation | ✅ Successful |
| Sampling Strategy | Top-p (Nucleus Sampling) |

> The final model successfully generated context-aware detective-style text using prompt-based inference.

---

## Demo Outputs

### Prompt 1

```text
Poirot looked at me and said
```

### Generated Output

```text
Poirot looked at me and said Poirot. I began to a few of the things were out of it. I was a small chest, and he had been across to them, and had out the good of the first of the road.
```

---

### Prompt 2

```text
Hastings, you must understand
```

### Generated Output

```text
Hastings, you must understand with a train in a situation. But the police were thrown across a small tent.
```

---

## Pipeline

```mermaid
flowchart TD

    A[Raw Dataset<br/>Poirot Investigates<br/>~53k words] --> B

    B[Text Cleaning & Normalisation<br/>Remove Gutenberg headers<br/>Fix quotes, dashes, spacing] --> C

    C[Tokenisation<br/>Regex word-level tokenisation<br/>Insert EOS tokens] --> D

    D[Vocabulary Construction<br/>word_to_id / id_to_word mappings<br/>Unknown token handling] --> E

    E[Numerical Encoding<br/>Convert words → integer token IDs] --> F

    F[Sequence Generation<br/>Create input-target pairs<br/>SEQ_LEN = 30] --> G

    G[LSTM Language Model Training] --> G1 & G2 & G3

    G1[Embedding Layer<br/>Dense word representations] --> H
    G2[Stacked LSTM Layers<br/>Sequence modelling] --> H
    G3[LayerNorm + Dropout<br/>Regularisation] --> H

    H[Training Optimisation<br/>Checkpoint saving<br/>Early stopping<br/>LR scheduling] --> I

    I[Text Generation<br/>Temperature scaling<br/>Top-p sampling<br/>Prompt-based inference] --> J

    J[Evaluation & Comparison<br/>Generated outputs<br/>LLM comparison<br/>Quality analysis]

    style A fill:#1F4E79,color:#fff,stroke:#1F4E79
    style B fill:#2E75B6,color:#fff,stroke:#2E75B6
    style C fill:#2E75B6,color:#fff,stroke:#2E75B6
    style D fill:#2E75B6,color:#fff,stroke:#2E75B6
    style E fill:#2E75B6,color:#fff,stroke:#2E75B6
    style F fill:#7D5A00,color:#fff,stroke:#7D5A00
    style G fill:#155724,color:#fff,stroke:#155724
    style G1 fill:#D4EDDA,color:#155724,stroke:#155724
    style G2 fill:#D4EDDA,color:#155724,stroke:#155724
    style G3 fill:#D4EDDA,color:#155724,stroke:#155724
    style H fill:#6F42C1,color:#fff,stroke:#6F42C1
    style I fill:#C05621,color:#fff,stroke:#C05621
    style J fill:#721C24,color:#fff,stroke:#721C24
```

---

## Model Architecture

The language model consists of:

- **Embedding Layer**
  - Learns dense vector representations for words

- **Stacked LSTM Layers**
  - Captures sequential dependencies and contextual patterns

- **Layer Normalisation**
  - Stabilises training

- **Dropout Regularisation**
  - Reduces overfitting

- **Dense Output Layer**
  - Predicts probability distribution across vocabulary

---

## Training Configuration

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| LSTM Units | 256 |
| Number of LSTM Layers | 2 |
| Batch Size | 64 |
| Sequence Length | 30 |
| Dropout | 0.3 |
| Optimiser | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 20 (Early Stopping Applied) |

---

## Text Generation Strategy

The model uses several decoding improvements to generate more natural text:

- **Temperature Scaling**
  - Controls randomness and creativity

- **Top-p (Nucleus) Sampling**
  - Samples only from the most probable tokens

- **Punctuation Limiting**
  - Prevents repetitive punctuation loops

- **EOS Token Stopping**
  - Allows cleaner sentence endings

---

## Comparison with Large Language Models

| Aspect | This Model | Large Language Models |
|---|---|---|
| Training Data | Single novel | Internet-scale datasets |
| Parameters | Small-scale | Billions of parameters |
| Coherence | Moderate | Very high |
| Grammar | Sometimes inconsistent | Highly accurate |
| Context Understanding | Limited | Strong |
| Style Mimicking | Good | Excellent |

### Key Insight

This project demonstrates how relatively small recurrent neural networks can still learn stylistic patterns and sentence structures from limited data, while modern transformer-based LLMs achieve superior coherence through massive datasets and scale.

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| NLP | Regex Tokenisation, Language Modelling |
| Data Processing | NumPy, JSON |
| Training | Google Colab |
| Visualisation | Matplotlib |
| Version Control | Git, GitHub |

---

## Project Structure

```text
.
├── notebook/
│   └── Language_Model_Poirot_Investigates.ipynb
│
├── checkpoints/
│   ├── best.weights.h5
│   └── config.json
│
├── outputs/
│   └── sample_outputs.txt
│
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/agatha-christie-language-model.git

cd agatha-christie-language-model

pip install -r requirements.txt
```

---

## Running the Project

### 1. Open the notebook

```bash
notebook/language_model.ipynb
```

### 2. Run all notebook cells sequentially

### 3. Enter a prompt during inference

Example:

```text
Poirot looked at me and said
```

---

## Model Checkpoints

The trained model weights are stored in:

```text
checkpoints/best.weights.h5
```

These checkpoints are required for inference and text generation.

---

## Key Technical Highlights

- Built a complete **word-level NLP pipeline** from preprocessing to inference
- Implemented custom **tokenisation and vocabulary encoding**
- Trained a stacked **LSTM language model** using TensorFlow/Keras
- Applied **early stopping, checkpointing, and learning rate scheduling**
- Implemented **top-p nucleus sampling** for higher-quality generation
- Compared generated outputs with transformer-based LLMs such as ChatGPT

---

## Future Improvements

- Replace LSTM with Transformer architecture
- Train on multiple Agatha Christie novels
- Add Beam Search decoding
- Deploy as a web application
- Fine-tune larger language models

---

## Author

AJ  
MSc Artificial Intelligence / Data Science

---
his project is for educational and research purposes.
