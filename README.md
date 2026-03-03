# Word2Vec in pure NumPy (Skip-gram with Negative Sampling)

This repository contains an educational implementation of the Word2Vec algorithm using the Skip-gram architecture with negative sampling, implemented **entirely in NumPy** (no PyTorch, TensorFlow, or other ML frameworks).

The focus of this project is on clarity and understanding:
- explicit forward pass
- loss computation
- gradient derivation
- parameter updates

It is intended as a reference for learning how Word2Vec works under the hood and as a basis for technical discussion in interviews or code reviews.

---

## Architecture

The model maintains two embedding matrices:

- **W1** → input (centre word) embeddings  
- **W2** → output (context word) embeddings  

Training pipeline:

1. Select a (centre, context) word pair  
2. Sample K negative words from a noise distribution  
3. Compute dot products  
4. Apply sigmoid  
5. Compute loss  
6. Manually compute gradients  
7. Update embeddings via SGD  

After training, the learned word vectors are taken from **W1**.

---

## Features

- Skip-gram architecture with negative sampling  
- Pure NumPy implementation  
- Manual gradient computation  
- Linear learning rate decay  
- Configurable hyperparameters:
  - embedding dimension
  - number of negative samples
  - learning rate
  - number of epochs
- Cosine similarity queries for nearest neighbours  

---


## Dataset

The model is trained on The Count of Monte Cristo (~460K words), a public domain text.

Preprocessing includes:

- Lowercasing  
- Removal of punctuation  
- Tokenization  
- Vocabulary construction  
- Generation of (centre, context) training pairs  

To keep training time manageable in pure NumPy, only the first *N* tokens of the corpus are used.

---


## Task description

This implementation addresses the following task:

> Implement the core training loop of Word2Vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimisation procedure (forward pass, loss, gradients, and parameter updates) for a standard Word2Vec variant (e.g. Skip-gram with negative sampling or CBOW).


---
