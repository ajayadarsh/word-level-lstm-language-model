---
title: "Word-Level Language Model - Poirot Investigates"
author: "Ajay Adarsh Sivakumar"
date: "19/02/2026"

---

# Word-Level LSTM Language Model  
### Text Generation in Agatha Christie Style

---

## Project Overview
This project implements a **word-level language model** using an LSTM network in TensorFlow. The model is trained on *"Poirot Investigates"* (~53,000 words) and is capable of generating text in a style similar to Agatha Christie.

Given a short prompt, the model predicts and generates the next sequence of words to form coherent sentences or paragraphs.

---

## Key Features
- Word-level tokenisation (not character-level)
- LSTM-based sequence modelling
- Nucleus (top-p) sampling for realistic text generation
- Temperature control for creativity vs coherence
- Punctuation control for cleaner outputs
- Early stopping and model checkpointing

---

## Model Architecture
The model is built using TensorFlow/Keras and consists of:

- **Embedding Layer** – Converts words into dense vector representations  
- **LSTM Layers** – Captures sequential dependencies in text  
- **Layer Normalisation + Dropout** – Improves stability and reduces overfitting  
- **Dense Output Layer** – Predicts probability distribution over vocabulary  

---

## Training Details

| Parameter        | Value        |
|----------------|-------------|
| Dataset         | Poirot Investigates |
| Tokens          | ~53,000 |
| Vocabulary Size | ~10k–12k |
| Sequence Length | 30 |
| Batch Size      | 64 |
| Embedding Dim   | 128 |
| LSTM Units      | 256 |
| Layers          | 2 |
| Dropout         | 0.3 |
| Optimiser       | Adam |
| Loss Function   | Sparse Categorical Crossentropy |

---

## Sample Outputs

**Prompt:**  
`Poirot looked at me and said`

**Generated:**  
`Poirot looked at me and said Poirot. I began to a few of the things were out of it. I was a small chest, and he had been across to them...`

---

**Prompt:**  
`Hastings, you must understand`

**Generated:**  
`Hastings, you must understand with a train in a situation. But the police were thrown across a small tent.`

---

## Comparison with Large Language Models

| Aspect        | This Model | Large Language Models |
|--------------|-----------|----------------------|
| Training Data | Single book | Massive internet-scale data |
| Coherence     | Moderate | Very high |
| Grammar       | Sometimes incorrect | Mostly correct |
| Creativity    | Limited | High |
| Context Awareness | Short-range | Long-range |

### Key Insight:
This model learns **local patterns and style**, while large language models understand **deep context and semantics** due to scale.

## How to Run

### Clone the repository

To test text generation **without retraining**:

1. Open the notebook.
2. Ensure the following files are in the same directory:
   - `best.weights.h5`
   - `vocab.json`
   - `config.json`
3. Run the following cells:
   - Configuration cell
   - Vocabulary loading cell
   - Model architecture definition cell
   - Load weights cell
   - Interactive generation cell
4. When prompted, enter a starting phrase such as: Poirot looked at me and said

5. The model will generate a continuation.
6. Type `quit` to exit interactive mode.

---

### How to Retrain the Model

If retraining is required:

1. Place the dataset file `61262-0.txt` in the working directory.
2. Run all notebook cells in order.
3. The best model checkpoint will automatically be saved as:best.weights.h5
