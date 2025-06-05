# 🧠 Comparison of Depression Detection Between LLM and Zero-Shot Learning Approaches

This project investigates and compares two NLP approaches—**fine-tuned Large Language Models (LLMs)** and **Zero-Shot Learning (ZSL)**—for detecting symptoms of depression in social media text. The models were trained and evaluated using the **Depression Analysis Dataset (DAD)**.

## 📌 Project Highlights

- **Fine-Tuned Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Zero-Shot Model**: BART (Bidirectional and Auto-Regressive Transformers)
- **Best Results**:  
  - **BERT**: 98.94% Accuracy | 99.24% F1-Score  
  - **BART**: 76.53% Accuracy | 81.26% F1-Score

The BERT model showed superior performance, making it well-suited for clinical or high-stakes applications. Meanwhile, the BART-based ZSL model offered reasonable performance with no additional training, making it ideal for low-resource environments or early screening tools.

## 🧪 Techniques & Tools Used

- **Frameworks**: PyTorch, TensorFlow
- **Concepts**: NLP preprocessing, EDA, model evaluation, hyperparameter tuning
- **Metrics**: Accuracy, F1-score, ROC-AUC

## 📄 Research Paper

This project was published in the **2025 IEEE International Colloquium on Signal Processing & Its Applications (CSPA)**.

🔗 [View Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/10933098)
