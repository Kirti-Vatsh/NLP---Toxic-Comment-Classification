# Toxic Comment Classification - Jigsaw Kaggle Challenge

A machine learning project to classify toxic comments in online discussions, using natural language processing techniques and deep learning models. This project was part of the Jigsaw Toxic Comment Classification Challenge on Kaggle and demonstrates advanced text analysis, feature engineering, and multi-class classification.

---

## 📌 Problem Statement

Online platforms face significant challenges in moderating toxic content. This project aims to build a robust model that can automatically classify comments into multiple categories:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

By doing so, we help platforms like Wikipedia and social media sites foster safer, more inclusive communities.

---

## 📊 Project Highlights

✅ Exploratory Data Analysis (EDA) to understand comment patterns, word distributions, and correlations  
✅ Feature engineering: word embeddings, cross-tab features, sentiment analysis  
✅ Model development using:
- Traditional machine learning (Logistic Regression, Random Forests)
- Deep learning (LSTMs, BiLSTMs, Transformers)
✅ Final model: Transformer-based multi-head attention architecture for multi-label classification  
✅ Achieved high validation accuracy and strong Kaggle leaderboard performance  

---

## 🧰 Tools & Technologies

| Category         | Tools & Libraries                                       |
|------------------|---------------------------------------------------------|
| Data Handling    | Pandas, NumPy                                           |
| NLP              | NLTK, spaCy, WordCloud                                  |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM                         |
| Deep Learning    | PyTorch, TensorFlow, HuggingFace Transformers           |
| Visualization    | Matplotlib, Seaborn, Plotly                             |
| Platform         | Kaggle, Jupyter Notebook                                |

---

## 📂 Files in This Repository

| File                                      | Description                                  |
|-------------------------------------------|----------------------------------------------|
| `ModernAI_Group19-4.pptx`                 | Final presentation summarizing the project   |
| `eda-toxic-comment-classification.ipynb`  | EDA notebook with insights and visualizations |
| `eda-toxic-comments-crosstab-and-other-features.ipynb` | Feature exploration notebook                   |
| `Analysis.ipynb`                          | Model analysis and evaluation                 |
| `eda.ipynb`                               | Initial EDA notebook (can be reviewed for redundancy) |
| `sentiment-analysis-on-toxic-comments.ipynb` | Final model notebook (Transformer-based)    |
| `sample_submission.csv`                   | Sample Kaggle submission                      |
| `test_labels.csv`                         | Kaggle test labels                            |
| `README.md`                               | Project overview (this file)                  |

---

## 🚀 Future Directions

- Fine-tune transformer models for specific classes
- Incorporate more contextual embeddings (e.g., ELMo, BERT)
- Build an interactive demo for comment classification
- Explore bias mitigation techniques in NLP models

---

## 🌟 Acknowledgments

- [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- Open-source NLP libraries and the Kaggle community

---

