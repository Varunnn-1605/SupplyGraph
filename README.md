<div align="center">

# ⚠️ GraphRisk: Supply Chain Disruption Prediction Engine

**A Graph Neural Network (GNN) framework for modeling cascading supply chain disruptions and risk propagation across FMCG operational networks.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.8.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#-overview) • [Key Features](#-key-features) • [Benchmark Results](#-benchmark-results) • [Architecture & Methodology](#-architecture--methodology) • [File Structure](#-file-structure) • [Installation & Usage](#-installation--usage) • [Citation](#-citation)

</div>

---

## 📌 Overview

In modern Fast-Moving Consumer Goods (FMCG) manufacturing, products do not exist in isolation. When products share physical production lines, temperature-controlled warehouses, or product classifications, operational failures trigger **cascading domino effects** across the entire supply chain.

Traditional tabular machine learning models (e.g. XGBoost) evaluate products independently, ignoring the relational topography of the network. **GraphRisk** models the supply chain as a graph:
- **Nodes**: Products / SKUs ($N = 40$).
- **Edges**: Shared operational dependencies across 4 relational dimensions (Manufacturing Plant, Storage Warehouse, Product Group, Product Sub-Group).
- **Model**: Inductive **GraphSAGE** architecture that aggregates neighborhood operational stresses while preserving node-specific volume dynamics.

Built upon the real-world **SupplyGraph** benchmark dataset, GraphRisk outperforms standard tabular baselines by over **17 percentage points in PR-AUC** and includes an interactive **Streamlit dashboard** with a live disruption simulator and BYOD ("Bring Your Own Data") engine.

---

## 🚀 Key Features

- **Interactive Disruption Simulator**: Stress-test supply chains by simulating a complete production halt (flatline state: $\text{mean}=0, \text{std}=0, \text{slope}=0$) at any product and monitoring risk spillover across 1-hop and 2-hop topological neighbors.
- **Continuous Risk Visualization**: Pyvis physics-driven network graph rendering continuous green-to-red risk scores, degree-scaled node sizing, and neighborhood disruption highlights.
- **Topological Density Toggle**: Switch dynamically between the **Dense Graph** ($w \ge 1$, 684 edges, 87.7% density) and the **Pruned Graph** ($w \ge 3$, 136 edges, 17.4% density, 6 clusters).
- **Leakage-Guarded Feature Engineering**: 18 clean, non-leaking temporal features engineered from 221-day time series across Production, Sales Orders, and Distributor Deliveries.
- **Bring Your Own Data (BYOD)**: Upload enterprise ERP extracts (Nodes, Edges, and Temporal CSVs) to automatically validate schemas, retrain GraphSAGE, and explore risk propagation interactively.

---

## 📊 Benchmark Results

Evaluated via strict **40-fold Leave-One-Out Cross-Validation (LOOCV)** with in-fold `StandardScaler` fitting (zero preprocessing leakage) on 40 FMCG products:

| Model Architecture | Graph Edges | Graph Density | ROC-AUC | PR-AUC (Key Metric) | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tabular Baseline)** | 0 | $0.0\%$ | 0.8400 | 0.7614 | **0.8182** | **0.9000** | **0.8571** |
| **GraphSAGE (Dense Graph, $w \ge 1$)** | 684 | $87.7\%$ | **0.9667** | **0.9310** | 0.8000 | 0.8000 | 0.8000 |
| **GraphSAGE (Pruned Graph, $w \ge 3$)** | 136 | $17.4\%$ | 0.9333 | 0.8591 | 0.7778 | 0.7000 | 0.7368 |

### Key Findings:
1. **GNN Superiority in Imbalanced Risk Ranking**: In supply chain risk prediction where disruptions are rare (25% positive rate), **PR-AUC is the gold standard**. Dense GraphSAGE delivers **0.9310 PR-AUC** vs. XGBoost\'s **0.7614** (+16.96% improvement).
2. **Graph Pruning Efficiency**: Pruning edges to $w \ge 3$ eliminates **80.1% of edges** (reducing density from $87.7\% \to 17.4\%$) while retaining **92.3% of the PR-AUC performance advantage over XGBoost**. High-confidence co-location (sharing $\ge 3$ facilities) carries the vast majority of risk signal.
3. **Over-Smoothing Immunity**: Standard GCN collapsed on this dense topology (ROC-AUC 0.7833, PR-AUC 0.4929) due to symmetric Laplacian feature averaging. GraphSAGE avoids this by isolating the central node\'s ego-features from neighbor aggregation.

---

## 🧠 Architecture & Methodology

```
┌─────────────────────────┐       ┌────────────────────────┐
│  221-Day Temporal Series│       │  Relational Databases  │
│(Production, Sales, Del.)│       │(Plant, Storage, Groups)│
└────────────┬────────────┘       └───────────┬────────────┘
             ▼                                ▼
┌─────────────────────────┐       ┌────────────────────────┐
│   Feature Engineering   │       │   Graph Construction   │
│  (18 Non-Leaking Feats) │       │ (Dense: 684 / Pruned: 136)│
└────────────┬────────────┘       └───────────┬────────────┘
             │                                │
             └───────────────┬────────────────┘
                             ▼
               ┌───────────────────────────┐
               │    SupplyGraphSAGE (2L)   │
               │  - SAGEConv(18 -> 16)     │
               │  - ReLU + Dropout(0.2)    │
               │  - SAGEConv(16 -> 1)      │
               └─────────────┬─────────────┘
                             ▼
               ┌───────────────────────────┐
               │ 40-Fold LOOCV Risk Scores │
               │   & Disruption Simulator  │
               └───────────────────────────┘
```

### 1. Data Cleaning & Validation
The raw dataset contains 41 entries with duplicate listing of `POP001L12P` across distinct production lines. The two temporal series are averaged across all 8 series to preserve physical scale, yielding a validated **40-node** graph.

### 2. Leakage Guard Policy
Proxy labels are derived from the 25th percentile largest single-day drop in Factory Issues ($\min_t \Delta y_t \le -2476.00$ units). To prevent catastrophic target leakage, **all 6 Factory Issue summary statistics are strictly quarantined and removed**, leaving 18 input features:
- Production (Unit & Weight): Mean, Std, Trend Slope (6 features)
- Sales Orders (Unit & Weight): Mean, Std, Trend Slope (6 features)
- Deliveries to Distributors (Unit & Weight): Mean, Std, Trend Slope (6 features)

### 3. Disruption Physics Consistency
When simulating a disruption at node $i$, setting only the feature mean to 0 while leaving standard deviation or slope positive creates an impossible physical contradiction (a zero-volume line with positive growth). GraphRisk sets **all three stats (mean, std, slope)** to $0.0$ simultaneously across production and delivery series, modeling a true flatline production halt.

---

## 📁 File Structure

```
SupplyGraph/
│
├── app.py                           # Streamlit interactive application (Frontend)
├── pipeline_core.py                 # Core computational engine (Graph, GNN, Pyvis, Sim)
│
├── supply_graph_construction.py     # Standalone Phase 0 pipeline script
├── supply_graph_construction.ipynb  # Executed Phase 0 notebook with outputs
├── phase1_benchmarking.py           # Standalone Phase 1 40-fold LOOCV benchmark script
├── phase1_benchmarking.ipynb        # Executed Phase 1 notebook with research narrative
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore configuration
├── LICENSE                          # Project license
├── README.md                        # Project documentation
│
└── Raw Dataset/Homogenoeus/         # Real-world FMCG dataset
    ├── Nodes/                       # Product definitions & plant/storage mappings
    │   ├── Nodes.csv
    │   ├── NodesIndex.csv
    │   ├── Node Types (Product Group and Subgroup).csv
    │   └── Nodes Type (Plant & Storage).csv
    ├── Edges/                       # Relational connection files
    │   ├── Edges (Plant).csv
    │   ├── Edges (Storage Location).csv
    │   ├── Edges (Product Group).csv
    │   ├── Edges (Product Sub-Group).csv
    │   └── EdgesIndex/
    └── Temporal Data/               # 221-day time series data
        ├── Unit/                    # Production, Sales, Delivery, Factory Issue (Units)
        └── Weight/                  # Production, Sales, Delivery, Factory Issue (MT)
```

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Varunnn-1605/SupplyGraph.git
cd SupplyGraph
```

### 2. Set Up Environment & Install Dependencies
Recommended: Python 3.10+
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the network, switch graph topologies, simulate disruptions, or upload custom enterprise datasets.

### 4. Run Scripts & Notebooks
- **Construct and validate graph**:
  ```bash
  python supply_graph_construction.py
  ```
- **Run the full 40-fold LOOCV benchmark**:
  ```bash
  python phase1_benchmarking.py
  ```

---

## 📚 Citation

If you use this dataset or benchmark in your research, please cite the original SupplyGraph benchmark paper:

```bibtex
@misc{wasi2024supplygraph,
      title={SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks}, 
      author={Azmine Toushik Wasi and MD Shafikul Islam and Adipto Raihan Akib},
      year={2024},
      eprint={2401.15299},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
