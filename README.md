# OncoGraph 🧬

**Pan-Cancer Subtype Classification and Patient Constellation via Graph Convolutional Networks (GCN)**

OncoGraph is a computational framework that uses Graph Neural Networks (GNNs) to classify cancer types and visualize patient similarity networks, built on the TCGA Pan-Cancer Atlas dataset (11,069 patients, 33 cancer types, 20,531 genes per patient).

Instead of treating each patient as an isolated data point (like traditional ML models such as SVM or Random Forest), OncoGraph builds a **Patient Constellation Graph** — a network where patients are connected to their 5 most molecularly similar neighbors (k-NN, cosine similarity). A GCN is then trained on this graph, allowing patients to "borrow" information from their genetic neighborhood — improving accuracy on rare cancer subtypes and enabling biologically interpretable predictions.

---

## ✨ Key Features

- **Pan-Cancer Classification** — Classifies patients into 1 of 33 cancer types from gene expression data alone.
- **Patient Similarity Graph** — Constructs a k-NN graph (k=5) over patients using cosine similarity of gene expression vectors.
- **Molecular Subtype Discovery** — Visualizes topological clustering to reveal intra-tumor heterogeneity.
- **Biomarker Identification** — Maps important model features back to HUGO gene symbols (e.g., `TP53`) for interpretability.
- **Prognostic Stratification** — Correlates patient clusters with clinical survival data (Overall Survival & PFI).
- **Anomaly / Open-Set Detection** — Flags predictions below a confidence threshold as "Unknown/Investigational" instead of forcing a misdiagnosis.

---

## 📊 Dataset

| | |
|---|---|
| **Name** | TCGA Pan-Cancer Atlas (PANCAN) — Gene Expression & Clinical Survival Data |
| **Source** | [UCSC Xena Browser](https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)) |
| **Reference** | The Cancer Genome Atlas Research Network, *Cell*, 2018 — DOI: [10.1016/j.cell.2018.03.022](https://doi.org/10.1016/j.cell.2018.03.022) |
| **Samples** | 11,069 patients across 33 cancer types |
| **Features** | 20,531 genes (Entrez IDs), batch-normalized, log₂ transformed RNA-Seq |
| **Targets** | Cancer type, Overall Survival (OS), Progression-Free Interval (PFI) |

> ⚠️ **Note:** The raw data files (e.g. the `.xena` gene expression matrix) are very large (over 1 GB) and are **not suitable for GitHub**. See [Handling Large Data Files](#-handling-large-data-files-important) below before pushing.

---

## 🗂 Project Structure

```
OncoGraph/
├── backend/
│   ├── analysis/          # Post-training analysis scripts (clustering, biomarkers, etc.)
│   ├── configs/           # Model & training configuration files
│   ├── data/               # (Local) processed/intermediate data — not pushed to GitHub
│   ├── evaluation/        # Evaluation utilities and metrics
│   ├── models/             # GCN model architecture definitions
│   ├── results/            # Output plots, metrics, saved artifacts
│   ├── training/           # Training loop / pipeline code
│   ├── utils/               # Shared helper functions
│   ├── evaluate_model.py   # Evaluate a trained model
│   ├── run_anomaly.py      # Open-set / anomaly (unknown subtype) detection
│   ├── run_biomarkers.py   # Biomarker (driver gene) identification
│   ├── run_subtypes.py     # Molecular subtype discovery / visualization
│   ├── run_survival.py     # Survival correlation analysis
│   ├── show_dataset.py     # Dataset inspection / summary
│   └── train_model.py      # Train the GCN
├── show_accuracy.py         # Quick accuracy check script
├── merged_data_sample.csv   # Small sample of the merged dataset
├── merged_sample.csv        # Small sample file
└── venv/                    # Local Python virtual environment — not pushed to GitHub
```

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/OncoGraph.git
cd OncoGraph

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> If you don't yet have a `requirements.txt`, generate one from your working environment with:
> ```bash
> pip freeze > requirements.txt
> ```

## ▶️ Usage

```bash
cd backend

# Inspect the dataset
python show_dataset.py

# Train the GCN
python train_model.py

# Evaluate the trained model
python evaluate_model.py

# Run downstream analyses
python run_subtypes.py       # Subtype discovery / visualization
python run_biomarkers.py     # Driver gene identification
python run_survival.py       # Survival correlation
python run_anomaly.py        # Open-set / unknown subtype detection
```

---

## 🧠 Methodology (Summary)

1. **Preprocessing** — Missing values imputed with 0.0; gene expression matrix merged with clinical survival data.
2. **Graph Construction** — Cosine similarity computed between all patients → converted to a k-NN adjacency matrix (k=5), forming the "Patient Constellation."
3. **Model** — A Graph Convolutional Network aggregates gene-expression features from each patient's local neighborhood to make predictions.
4. **Interpretability** — Important features (genes) are mapped from Entrez IDs to HUGO gene symbols via the `mygene` API.
5. **Safety Layer** — A softmax confidence threshold (<60%) flags low-confidence predictions as "Unknown/Investigational" rather than forcing a diagnosis.

---

## 📁 Handling Large Data Files (Important)

GitHub blocks files over 100 MB by default, and your dataset folder contains files over 1 GB (e.g. the `.xena` RNA-Seq matrix and the supplemental survival table). **Do not commit these directly.** Before uploading:

1. Create a `.gitignore` file in the project root (see below) so `venv/`, large raw data files, and the `data/` folder are never staged.
2. If collaborators need the raw data, share the download link instead (already documented above under Dataset), or use **Git LFS** ([git-lfs.com](https://git-lfs.com)) if you must version large files.

**Suggested `.gitignore`:**
```
venv/
__pycache__/
*.pyc
backend/data/
*.xena
*.xena*
EB++AdjustPANCAN*
Survival_SupplementalTable*
.env
.DS_Store
```

---

## 👥 Team

| Registration No. | Name |
|---|---|
| 23BCE1863 | Byna Sriroop |
| 23BCE1613 | Gunnam Reddy Sujith Reddy |
| 23BCE1236 | Malli Mohith |

---

## 📚 Key References

- The Cancer Genome Atlas Research Network. (2018). *Cell*. [10.1016/j.cell.2018.03.022](https://doi.org/10.1016/j.cell.2018.03.022)
- Ozdemir, C., Vashishath, Y., & Bozdag, S. (2025). IGCN: Integrative graph convolution networks for patient level insights and biomarker discovery. *Bioinformatics*. [10.1093/bioinformatics/btaf313](https://doi.org/10.1093/bioinformatics/btaf313)
- Li, B., Wang, T., & Nabavi, S. (2021). Cancer molecular subtype classification by graph convolutional networks on multi-omics data. *ACM-BCB*. [10.1145/3459930.3469542](https://doi.org/10.1145/3459930.3469542)
- Li, X. et al. (2022). MoGCN: A multi-omics integration method based on GCN for cancer subtype analysis. *Frontiers in Genetics*. [10.3389/fgene.2022.806842](https://doi.org/10.3389/fgene.2022.806842)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
