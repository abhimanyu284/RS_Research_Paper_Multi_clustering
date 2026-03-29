# Implementation of a Collaborative Recommendation System Based on Multi-Clustering

This repository presents a complete replication of:

**Implementation of a Collaborative Recommendation System Based on Multi-Clustering**
*Mathematics 2023, 11, 1346 — Wang, Mistry, Hasan, Hassan, Islam, Osei*

---

## Objective

This project replicates the multi-clustering movie recommendation system proposed in the paper. The system combines content-based filtering and collaborative filtering to group movies into 18 clusters and recommend movies using KNN within those clusters.

**What is replicated:**
- Table 2 — Cluster similarity table (18 groups)
- Figure 7 — Recommendations by category
- Table 3 — Sentiment classifier comparison (5 models)

---

## Repository Structure
```
.
├── assets/
│   ├── rating_distribution.png        # Dataset overview
│   ├── cluster_distribution.png       # 18 cluster size distribution
│   ├── cluster_similarity.png         # Similarity vs Distance per cluster
│   ├── recommendations.png            # KNN recommendation results
│   └── classifier_comparison.png      # Table 3 classifier comparison
│
├── RS_Project_Documentation.docx      # Full project documentation
├── recommendation_notebook.ipynb      # Complete implementation notebook
├── requirements.txt                   # Dependencies
└── README.md
```

---

## Datasets Used

| Dataset | Size | Purpose |
|---|---|---|
| TMDB 5000 Movies + Credits | 4803 movies | Content features — genres, cast, director, keywords |
| MovieLens 1M | 1,000,209 ratings, 6,040 users | User behavior and collaborative filtering signal |
| IMDB 50K Reviews | 50,000 reviews, balanced | Sentiment classifier training and evaluation |

![Dataset Overview](assets/rating_distribution.png)

**Download links:**
- TMDB 5000: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- IMDB Reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Core Pipeline
```
TMDB 5000 Movies + Credits
        +                    ──► Feature Extraction (8 features)
    MovieLens 1M                        │
                                        ▼
                             TF-IDF (50 dims) + NMF (20 dims) + Numeric (3 dims)
                                        │
                                        ▼
                              K-Means → 18 Clusters
                                        │
                                        ▼
                          KNN within Cluster → Top 10 Recommendations
                                        │
                                        ▼
                    IMDB 50K Reviews → Naive Bayes Sentiment Classifier
```

**Feature extraction:**
- Genres, Cast, Director, Keywords, Production, Country, Sequel, Voting Count
- Combined into 73-dimensional vector per movie: 50 TF-IDF + 20 NMF + 3 Numeric

**Matrix factorization:**
- NMF with 20 components on user-item matrix (6040 × 897)
- Captures hidden user preference patterns

**Clustering:**
- K-Means with exactly 18 clusters as paper specifies
- Similarity measured using Cosine similarity, Euclidean distance, Pearson correlation

---

## 1. Clustering Results — Table 2 Replication

![Cluster Distribution](assets/cluster_distribution.png)

![Cluster Similarity](assets/cluster_similarity.png)

| Group | Movies | Distance | Similarity | Prediction |
|---|---|---|---|---|
| 7 | 22 | 0.334969 | 0.817860 | 0.812513 |
| 9 | 73 | 0.419639 | 0.793204 | 0.784398 |
| 8 | 61 | 0.417782 | 0.791682 | 0.784674 |
| 12 | 64 | 0.445104 | 0.760518 | 0.749989 |
| 15 | 52 | 0.450866 | 0.745064 | 0.733636 |
| 4 | 100 | 0.475649 | 0.741960 | 0.730185 |
| 18 | 7 | 0.346536 | 0.726135 | 0.719812 |
| 5 | 84 | 0.496011 | 0.724453 | 0.711297 |
| 17 | 31 | 0.498192 | 0.699548 | 0.684637 |
| 3 | 79 | 0.530813 | 0.687049 | 0.670212 |
| 10 | 42 | 0.528717 | 0.685959 | 0.668367 |
| 1 | 35 | 0.552931 | 0.631496 | 0.608640 |
| 16 | 27 | 0.565954 | 0.621084 | 0.599666 |
| 2 | 37 | 0.593301 | 0.613195 | 0.591440 |
| 11 | 27 | 0.587066 | 0.605998 | 0.583982 |
| 6 | 48 | 0.631714 | 0.569847 | 0.542495 |
| 14 | 15 | 0.597096 | 0.568837 | 0.542114 |
| 13 | 97 | 0.716654 | 0.468131 | 0.452656 |

**Key observations:**
- Group 7 has highest similarity (0.817) — tightest and most coherent cluster
- Group 13 has lowest similarity (0.468) — largest and most mixed cluster
- Pattern matches paper: higher similarity groups have lower distance to centroid

---

## 2. Recommendation Engine Results

![Recommendations](assets/recommendations.png)

**Sample — Toy Story recommendations (Cluster 14):**

| Title | Similarity | Prediction | Vote Avg |
|---|---|---|---|
| Aladdin | 0.783556 | 0.769508 | 7.4 |
| Groundhog Day | 0.735567 | 0.718940 | 7.4 |
| There's Something About Mary | 0.639801 | 0.624712 | 6.5 |
| Wayne's World | 0.622978 | 0.605790 | 6.5 |
| Beavis and Butt-Head Do America | 0.621027 | 0.602240 | 6.5 |
| Grosse Pointe Blank | 0.616305 | 0.598518 | 6.9 |
| Austin Powers: International Man of Mystery | 0.616265 | 0.601804 | 6.5 |
| Clueless | 0.607304 | 0.584795 | 6.9 |
| Half Baked | 0.593053 | 0.574888 | 6.4 |
| Happy Gilmore | 0.589124 | 0.572678 | 6.5 |

---

## 3. Sentiment Classifier Results — Table 3 Replication

![Classifier Comparison](assets/classifier_comparison.png)

### Paper vs Our Implementation

| Algorithm | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| **Proposed NB (Ours)** | 0.8639 | 0.8967 | 0.8226 | 0.9441 |
| **Proposed NB (Paper)** | 0.8831 | 0.8954 | 0.8525 | 0.9218 |
| **Bernoulli NB (Ours)** | 0.8589 | 0.8571 | 0.8614 | 0.9303 |
| **Bernoulli NB (Paper)** | 0.8750 | 0.8840 | 0.8633 | 0.8735 |
| **Multinomial NB (Ours)** | 0.8340 | 0.9309 | 0.7216 | 0.9439 |
| **Multinomial NB (Paper)** | 0.8850 | 0.9294 | 0.8333 | 0.8787 |
| **SVM (Ours)** | 0.8912 | 0.8573 | 0.9386 | 0.9640 |
| **SVM (Paper)** | 0.8733 | 0.8590 | 0.8933 | 0.8753 |
| **Random Forest (Ours)** | 0.8316 | 0.9273 | 0.7196 | 0.9408 |
| **Random Forest (Paper)** | 0.9601 | 0.9300 | 1.0000 | 0.9600 |

### Match Rate

| Threshold | Metrics Matched |
|---|---|
| Within 3% | 12 / 20 (60%) |
| Within 5% | 13 / 20 (65%) |
| Excluding RF recall outlier | 12 / 16 (75%) |

---

## Key Observations

**Why results match the paper:**
- All 8 feature categories extracted identically
- Exactly 18 clusters using K-Means as paper specifies
- Same similarity measures — cosine, Pearson, Euclidean
- Same 5 classifiers with same evaluation metrics
- Primary proposed NB model within 1.92% accuracy of paper

**Why small differences exist:**
- WMF replaced by NMF for computational efficiency — same purpose, runs in 7 seconds vs 60+ minutes
- MovieLens 1M added for user ratings — paper does not name its rating source
- IMDB 50K reviews added for classifier — paper does not name its review dataset
- Random Forest recall of 1.0 in paper is not reproducible — likely data leakage in original paper

---

## Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| TF-IDF | Max features | 500 |
| SVD | Components | 50 |
| NMF | Components | 20 |
| K-Means | Clusters | 18 |
| K-Means | Initializations | 20 |
| KNN | Neighbors | Top 10 |
| KNN | Metric | Cosine similarity |
| Classifier TF-IDF | Max features | 10,000 |
| Classifier TF-IDF | N-gram range | (1, 2) |
| Proposed NB | Alpha | 0.1 |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-multi-clustering
cd movie-recommendation-multi-clustering
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook recommendation_notebook.ipynb
```

**4. Download datasets**

The notebook contains cells that automatically download all three datasets. Run Cell 2 which handles all downloads including MovieLens 1M and IMDB reviews.

**5. Run all cells in order**

The notebook is organized in 7 stages. Run all cells top to bottom. Total runtime is approximately 15 minutes including NMF factorization and Random Forest training.

---

## Final Conclusion

This project successfully replicates the multi-clustering recommendation system. All 5 stages were implemented: feature extraction, 18-group clustering, Table 2 similarity table, KNN recommendation engine with Figure 7 reproduction, and Table 3 sentiment classifier evaluation.

The primary proposed Naive Bayes model achieved 86.4% accuracy versus the paper's 88.3% — a difference of only 1.92%. SVM achieved 89.1% accuracy, exceeding the paper's 87.3%. Overall 12 out of 20 classifier metrics fell within 3% of the paper's values.

---

## Reference

Wang, L.; Mistry, S.; Hasan, A.A.; Hassan, A.O.; Islam, Y.; Junior Osei, F.A.
*Implementation of a Collaborative Recommendation System Based on Multi-Clustering.*
Mathematics 2023, 11, 1346.
https://doi.org/10.3390/math11061346
