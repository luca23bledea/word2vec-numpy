# Word2Vec in Pure NumPy  
### Skip-gram with Negative Sampling (SGNS)

This repository contains an educational implementation of the **Word2Vec** algorithm using the **Skip-gram architecture with Negative Sampling**, implemented entirely in **NumPy**.

- No PyTorch  
- No TensorFlow  
- No automatic differentiation  

All gradients and updates are derived and implemented manually.

The purpose of this project is to deeply understand how Word2Vec works internally and to demonstrate the full optimisation pipeline:
- Forward pass  
- Loss computation  
- Manual gradient derivation  
- Parameter updates with SGD  

---

## Mathematical Objective

For a center word \( w_c \) and a context word \( w_o \), Skip-gram with negative sampling maximizes:

\[
\log \sigma(v_{w_o}^T v_{w_c}) +
\sum_{k=1}^{K} \log \sigma(-v_{w_k}^T v_{w_c})
\]

Where:

- \( \sigma(x) \) is the sigmoid function  
- \( v_{w_c} \) is the center word embedding  
- \( v_{w_o} \) is the positive context embedding  
- \( w_k \) are negatively sampled words  
- \( K \) is the number of negative samples  

Instead of computing a full softmax over the vocabulary (cost \(O(V)\)), negative sampling reduces the complexity to \(O(K)\), where \(K \ll V\).

---

## Architecture

The model maintains two embedding matrices:

- **W1** → input (center word) embeddings  
- **W2** → output (context word) embeddings  

Training pipeline:

1. Select a (center, context) word pair  
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
- Cosine similarity queries for nearest neighbors  

---

## Dataset

The model is trained on a cleaned subset of a public domain literary text 

Preprocessing includes:

- Lowercasing  
- Removal of punctuation  
- Tokenization  
- Vocabulary construction  
- Generation of (center, context) training pairs  

To keep training time manageable in pure NumPy, only the first *N* tokens of the corpus are used.

---