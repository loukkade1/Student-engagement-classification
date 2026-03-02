# 🎓 Online Course Engagement — Classification

> Predicting whether students will complete an online course based on engagement behavior.

**Course:** MSCA 691 — Advanced Data Mining | Concordia University 

**Team:** Sarb, Helen, Sunicha

---

## 📋 Project Overview

Online learning platforms face significant student dropout rates. This project builds classification models to predict course completion using student engagement metrics — enabling early intervention and personalized learning experiences.

**Task:** Binary Classification (Completed vs. Not Completed)  
**Dataset:** [Online Course Engagement Data]([https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset]) — ~9,000 student records, 9 columns  
**Best Model:** Support Vector Classification (RBF kernel + hyperparameter tuning) — **86.89% test accuracy**

---

## 📁 Repository Structure

```
├── Assingment_1_Final.ipynb    # Full analysis notebook (EDA, preprocessing, modeling)
├── Assignment1_MSCA_691.pdf    # Presentation slides
├── data/
│   └── online_course_engagement_data.csv
├── README.md
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| CourseCategory | Categorical | Arts, Health, Science, Programming, Business |
| TimeSpentOnCourse | Numerical | Total time spent (continuous) |
| NumberOfVideosWatched | Numerical | Videos viewed count |
| NumberOfQuizzesTaken | Numerical | Quizzes attempted |
| QuizScores | Numerical | Average quiz score |
| CompletionRate | Numerical | % of course completed |
| DeviceType | Binary | 0 = Desktop, 1 = Mobile |
| **CourseCompletion** | **Target** | **0 = Not Completed, 1 = Completed** |

**After cleaning:** 8,123 rows (877 duplicates removed), 0 missing values  
**Target split:** 56% Not Completed / 44% Completed

---

## 🔧 Methodology

### Pre-processing
- Dropped `UserID` (identifier only)
- **StandardScaler** on 5 numerical features (no log/sqrt needed — uniform distribution)
- **One-hot encoded** CourseCategory (m−1 = 4 dummies, Arts as base)
- Train/Test split: 80/20 (random_state=32)
- Final feature matrix: **10 IVs + 1 DV**

### Models Evaluated

We compared **4 classifier types** across 7 total model runs:

| Model | Approach | CV Accuracy | Test Accuracy | F1-Score | Runtime |
|---|---|---|---|---|---|
| Logistic Regression (base) | Linear | 79.33% | 78.22% | 76.14% | 0.72s |
| Logistic Regression + HPT | Linear | 79.22% | 78.50% | 77.13% | 0.91s |
| Gaussian Naïve Bayes | Probabilistic | 82.46% | 81.97% | 79.14% | 0.24s |
| Gaussian Process (RBF) | Kernel | **89.14%** | **88.60%** | **87.41%** | 1453s |
| SVC — Linear | Kernel | 79.39% | — | 76.16% | — |
| SVC — RBF | Kernel | 85.93% | 85.97% | 83.61% | — |
| **SVC — RBF + HPT** | **Kernel** | **87.66%** | **86.89%** | **85.85%** | **38.7s** |

### Why SVC (RBF + HPT) as Final Model?
- Near-GPC accuracy (86.89% vs 88.60%) but **37× faster** (39s vs 1,453s)
- **Lowest variance** across folds (0.79% STD)
- Best **balance** between precision (86.49%) and recall (85.25%)
- Scalable to larger datasets

---

## 📈 Key Findings

1. **Low inter-feature correlation** — no multicollinearity; favorable for Naïve Bayes
2. **Top predictors** (from OLS): CompletionRate (0.159), QuizScores (0.155), NumberOfQuizzesTaken (0.146)
3. **DeviceType is not significant** — desktop vs mobile doesn't predict completion
4. **Non-linear models outperform linear ones** — RBF kernel captures complex engagement patterns
5. **GPC achieves highest accuracy but is impractical** — O(n³) complexity limits scalability

---

## 🛠️ Tech Stack

- **Python**
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (LogisticRegression, GaussianProcessClassifier, GaussianNB, SVC)
- statsmodels (OLS regression)

---

## 📄 License

This project was completed as part of academic coursework at the Concordia University. Dataset sourced from Kaggle.
