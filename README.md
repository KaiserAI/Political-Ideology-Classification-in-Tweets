# 🗳️ Political Ideology Classification in Tweets
### NLP Analysis of Ideological Stance in Spanish Social Media

![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-Transformers_%2F_BETO-yellow)
![Library](https://img.shields.io/badge/Lib-PyTorch_%7C_Scikit--Learn-EE4C2C)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **📄 [Read the Full Technical Report (PDF)](./docs/report.pdf)**
> *A comprehensive analysis including theoretical background, full architectural details, and in-depth error analysis.*

---

## 📋 Project Overview
Social media platforms generate massive amounts of political discourse daily. Analyzing this data manually is unfeasible. This project develops an automated **Natural Language Processing (NLP)** pipeline to classify the ideological stance of Spanish tweets.

The system analyzes linguistic patterns to categorize text into four distinct segments: **Left, Moderate Left, Moderate Right, and Right**.

### Key Objectives
* **Data Engineering:** Implementation of a robust preprocessing pipeline to handle social media noise (slang, emojis, mentions).
* **Benchmarking:** Rigorous comparison between Classical Machine Learning models and Deep Learning architectures.
* **SOTA Adaptation:** Fine-tuning the **BETO (Spanish BERT)** Transformer model to capture semantic nuance and context.

---

## ⚙️ Methodology

The project follows a structured data science lifecycle, detailed in *Chapter 3 & 4* of the attached report.

### 1. Data Processing
Using the `politicES_phase_2` dataset (Hugging Face), the raw text undergoes a strict cleaning process:
* **Normalization:** Lowercasing and removal of special characters.
* **Noise Removal:** Regex-based filtering of URLs, hashtags (`#`), and user mentions (`@`).
* **Linguistic Processing:** Tokenization, Stop-word removal, and Lemmatization to reduce feature dimensionality.

### 2. Model Architecture
I engineered and evaluated three tiers of models to find the optimal performance trade-off:

| Tier | Models Implemented | Description |
| :--- | :--- | :--- |
| **1. Baselines** | Naive Bayes, SVM, Random Forest | Statistical approaches using **TF-IDF** vectorization. |
| **2. Deep Learning** | Bi-LSTM | Bidirectional Long Short-Term Memory network with embedding layers to capture sequence order. |
| **3. SOTA** | **BETO (Transformers)** | Transfer learning using the pre-trained Spanish BERT model, fine-tuned for sequence classification. |

---

## 📊 Experimental Results

*For detailed confusion matrices and error analysis, please refer to Chapter 5 of the report.*

The experiments demonstrated that Contextual Embeddings (Transformers) significantly outperform statistical methods in detecting political nuance.

### Performance Summary (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | XX.X% | 0.XX | 0.XX | 0.XX |
| **SVM (Linear)** | XX.X% | 0.XX | 0.XX | 0.XX |
| **Bi-LSTM** | XX.X% | 0.XX | 0.XX | 0.XX |
| **BETO (Ours)** | **XX.X%** | **0.XX** | **0.XX** | **0.XX** 🏆 |

> **Technical Insight:** Classical models (SVM) performed well on explicit keywords, but struggled with irony and context. The **BETO** model resolved these ambiguities by analyzing the entire sentence structure via its attention mechanisms.

---

## 📂 Project Structure

```text
/
├── /docs
│    └── report.pdf             # Full Technical Report (LaTeX)
├── /src
│    ├── /preprocessing         # Scripts for data cleaning & tokenization
│    ├── /models                # Model definitions (Classical & Deep Learning)
│    └── /training              # Training loops and evaluation scripts
├── /notebooks                  # Jupyter Notebooks for EDA and visualization
└── README.md
