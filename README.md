# Word2Vec in pure NumPy (Skip-gram with Negative Sampling)

This repository contains an educational implementation of the Word2Vec algorithm using the Skip-gram architecture with negative sampling, implemented **entirely in NumPy** (no PyTorch, TensorFlow, or other ML frameworks).

The focus of this project is on clarity and understanding:
- explicit forward pass
- loss computation
- gradient derivation
- parameter updates

It is intended as a reference for learning how Word2Vec works under the hood and as a basis for technical discussion in interviews or code reviews.

---

## Features

- **Skip-gram architecture** with negative sampling
- **Pure NumPy implementation** (no deep learning frameworks)
- **Configurable hyperparameters**:
  - embedding dimension
  - number of negative samples
  - learning rate (with linear decay)
  - number of epochs
  - context window size (if implemented)
- Simple **training loop** with loss logging
- Basic **similarity queries** (e.g., most similar words)

---

## Task description

This implementation addresses the following task:

> Implement the core training loop of Word2Vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard Word2Vec variant (e.g. Skip-gram with negative sampling or CBOW).


---
