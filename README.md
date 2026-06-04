# Number of Clusters Discovery

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![UMAP](https://img.shields.io/badge/UMAP-Dimensionality%20Reduction-purple)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?logo=python)
![Platform](https://img.shields.io/badge/Platform-Jupyter-yellow?logo=jupyter)
![Data](https://img.shields.io/badge/Data-text8-lightgrey)

Discovers the optimal number of semantic word clusters in the [text8](http://mattmahoney.net/dc/textdata.html) corpus using co-occurrence embeddings, dimensionality reduction, and K-means clustering.

---

## Pipeline

```
text8 (17M tokens)
  → Punkt tokenisation + stopword removal + length filter (3–18 chars)
  → Lemmatisation + POS-aware synonym mapping
  → Frequency filter (50–5,000 occurrences) → 9,265 unique words
  → Co-occurrence matrix (window = 5) + L2 normalisation  [9265 × 9265]
  → PCA (20 components)
  → UMAP (5 components, n_neighbors=10, min_dist=0.1)
  → K-means (k = 2 … 19) evaluated by Silhouette Score on validation set
  → Optimal k = 7
```

**Split:** 64 % train / 16 % validation / 20 % test

---

## Results

| K | Validation Silhouette | Train Inertia |
|---|----------------------|---------------|
| 3 | 0.611 | 40213.301 |
| 4 | 0.632 | 33751.602 |
| 5 | 0.653 | 29646.887 |
| 6 | 0.661 | 25722.094 |
| **7** | **0.698** | **23330.656** |

Test Silhouette Score at k = 7: **0.633**

K = 7 was selected via Occam's Razor — the silhouette score plateaus after this point, and higher k adds complexity without meaningful gain.

---

## Setup

### 1. Get the data

Download [text8](http://mattmahoney.net/dc/textdata.html) and place the extracted `text8` file in the repo root.

```bash
curl -O http://mattmahoney.net/dc/text8.zip
unzip text8.zip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK data is downloaded automatically on first run.

### 3. Run

Open `kmeans_clustering.ipynb` in Jupyter and run all cells.

```bash
jupyter notebook kmeans_clustering.ipynb
```

> **Note:** Originally developed on Google Colab. The Drive mount has been removed; the notebook reads `text8` from the repo root directly.

---

## File Structure

```
.
├── kmeans_clustering.ipynb   # Main notebook
├── report.pdf                # Write-up
├── requirements.txt
└── text8                     # Dataset (not tracked — ~96 MB, add manually)
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `nltk` | Punkt tokeniser, stopwords, WordNet lemmatiser |
| `umap-learn` | UMAP dimensionality reduction |
| `scikit-learn` | PCA, K-means, Silhouette Score |
| `scipy` | Sparse co-occurrence matrix |
| `tqdm` | Progress bars |
