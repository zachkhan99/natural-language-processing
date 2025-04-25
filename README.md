# Natural Language Processing Practice
**Author:** Zach Khan

## Overview  
This assignment explores three key natural language processing (NLP) tasks:

1. **Text Classification using 1D Convolutional Neural Networks (CNNs)**
2. **Part-of-Speech (POS) Tagging using Viterbi HMM and Unigram Tagger**
3. **Text Generation using RNN-based Models**

Each task includes model experimentation, performance evaluation, and result analysis.

---

## 1D CNN for Text Classification

### Goal  
Classify movie reviews from the IMDB dataset using a 1D CNN model.

### Models Compared  
- **Default Model (Filter Sizes [3, 4, 5])**
  - **Test Accuracy:** 86.28%
  - **Test Loss:** 0.4594
  - **Training Accuracy:** 98.21%
  - **Validation Accuracy:** 88.44%
  - **Notes:** Early stopping after epoch 4 due to overfitting

- **Alternative Model (Filter Sizes [2, 3, 4])**
  - **Test Accuracy:** 85.36%
  - **Test Loss:** 0.4368
  - **Training Accuracy:** 97.74%
  - **Validation Accuracy:** 86.64%
  - **Notes:** Slightly less overfitting, same early stopping behavior

### Key Takeaways  
The model using [3, 4, 5] filter sizes performed slightly better in terms of accuracy, while the [2, 3, 4] model had a lower loss and overfit less. Larger filters helped capture longer word patterns, which seemed important for sentiment classification.

---

## POS Tagging using Viterbi HMM

### Models Compared  
- **Viterbi HMM Tagger**
  - **Accuracy:** 47.37%
  - **Processing Time:** 7.44 seconds
  - **Notes:** Overflow issues, poor performance due to numerical instability

- **Unigram Tagger**
  - **Accuracy:** 88.27%
  - **Processing Time:** 0.02 seconds
  - **Notes:** Much simpler, fast, and surprisingly accurate for the dataset

### Key Takeaways  
The Unigram Tagger outperformed the HMM both in speed and accuracy. Despite the HMM’s theoretical strengths in handling sequences, it underperformed likely due to implementation or data sparsity issues. Improving HMM performance would require addressing overflow errors and possibly using backoff strategies or smoothing.

---

## RNN-based Text Generation

### Models Compared  
- **Vanilla RNN**
  - **Validation Loss:** 2.1678
  - **Training Time:** ~37s (5 epochs)
  - **Generated Text:** Some structure, many spelling/nonsense errors

- **Stacked RNN**
  - **Validation Loss:** 2.0813
  - **Training Time:** ~48s (5 epochs)
  - **Generated Text:** Slightly more coherent

- **Bidirectional RNN**
  - **Validation Loss:** 1.9487
  - **Training Time:** ~118s (5 epochs)
  - **Generated Text:** Most structured, varied, still error-prone

### Key Takeaways  
The Bidirectional RNN performed best in terms of loss and output quality, albeit with the longest training time. All models mimicked Shakespearean formatting, but none generated fully coherent text. With a small dataset and fewer epochs, there's a trade-off between runtime and quality.

---

## Final Notes  
Across all tasks, model complexity tended to improve performance but increased computational cost. Simpler models like the Unigram Tagger proved effective in specific contexts, while deeper models like Bidirectional RNNs offered the most nuanced results for generation tasks.

