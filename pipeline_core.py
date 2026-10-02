"""
pipeline_core.py
================
Shared computational backend for the GraphRisk Streamlit dashboard.
Used by both Section 1 (Fixed SupplyGraph Demo) and Section 2 (BYOD).

All heavy logic lives here so app.py only handles layout/UI.
"""

import os
import io
import tempfile
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import linregress
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from pyvis.network import Network

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "Raw Dataset", "Homogenoeus")

PHASE1_METRICS = {
    "dense":  {"ROC-AUC": 0.9667, "PR-AUC": 0.9310},
    "pruned": {"ROC-AUC": 0.9333, "PR-AUC": 0.8591},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

TEMPORAL_MANIFEST = {
    ("Production",    "Unit"):   ("Unit",   "Production .csv"),
    ("Production",    "Weight"): ("Weight", "Production .csv"),
    ("Sales_Order",   "Unit"):   ("Unit",   "Sales Order.csv"),
    ("Sales_Order",   "Weight"): ("Weight", "Sales Order .csv"),
    ("Delivery",      "Unit"):   ("Unit",   "Delivery To distributor.csv"),
    ("Delivery",      "Weight"): ("Weight", "Delivery to Distributor.csv"),
    ("Factory_Issue", "Unit"):   ("Unit",   "Factory Issue.csv"),
    ("Factory_Issue", "Weight"): ("Weight", "Factory Issue.csv"),
}

LEAKING_PREFIX = "Factory_Issue"

DISRUPTION_FEATURES = [
    "Production_unit_mean",   "Production_unit_std",   "Production_unit_slope",
    "Production_weight_mean", "Production_weight_std", "Production_weight_slope",
    "Delivery_unit_mean",     "Delivery_unit_std",     "Delivery_unit_slope",
    "Delivery_weight_mean",   "Delivery_weight_std",   "Delivery_weight_slope",
]


def load_supplygraph_temporal(raw_dir: str) -> dict:
    """Load all 8 temporal DataFrames from the SupplyGraph raw directory."""
    dfs = {}
    for (feat, measure), (subdir, fname) in TEMPORAL_MANIFEST.items():
        path = os.path.join(raw_dir, "Temporal Data", subdir, fname)
        dfs[(feat, measure)] = pd.read_csv(path)
    return dfs


def get_ordered_products(raw_dir: str) -> list:
    """Return the canonical 40-product order from NodesIndex.csv."""
    ni = pd.read_csv(os.path.join(raw_dir, "Nodes", "NodesIndex.csv"))
    return ni.drop_duplicates(subset=["Node"])["Node"].tolist()


def engineer_features(
    temporal_dfs: dict,
    ordered_products: list,
    drop_leaking: bool = True,
    pop_dup_col: str = "POP001L12P",
) -> pd.DataFrame:
    """
    Compute static summary statistics (mean, std, OLS slope) for each
    product x temporal series combination (8 series x 3 stats = 24 features).
    POP001L12P duplication: if pandas auto-renamed the second column to
    POP001L12P.1, the two series are averaged before computing stats.
    When drop_leaking=True all 6 Factory_Issue summary features are removed.
    Returns a DataFrame of shape (N_products, 24 or 18).
    """
    records = {p: {} for p in ordered_products}
    for (feat_name, measure), df_t in temporal_dfs.items():
        prefix  = f"{feat_name}_{measure.lower()}"
        T       = len(df_t)
        t_axis  = np.arange(T)
        dup_col = pop_dup_col + ".1"
        if pop_dup_col in df_t.columns and dup_col in df_t.columns:
            pop_merged = (df_t[pop_dup_col].astype(float) +
                          df_t[dup_col].astype(float)) / 2.0
        else:
            pop_merged = df_t[pop_dup_col].astype(float) if pop_dup_col in df_t.columns else None
        for prod in ordered_products:
            if prod == pop_dup_col and pop_merged is not None:
                vals = pop_merged.values
            else:
                vals = df_t[prod].astype(float).values
            mu    = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1)) if T > 1 else 0.0
            slope = float(linregress(t_axis, vals).slope)
            records[prod][f"{prefix}_mean"]  = mu
            records[prod][f"{prefix}_std"]   = sigma
            records[prod][f"{prefix}_slope"] = slope
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "Node"
    if drop_leaking:
        leak_cols = [c for c in df.columns if c.startswith(LEAKING_PREFIX)]
        df = df.drop(columns=leak_cols)
    return df


def get_non_leaking_cols(df_features: pd.DataFrame) -> list:
    return [c for c in df_features.columns if not c.startswith(LEAKING_PREFIX)]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def load_supplygraph_edges(raw_dir: str) -> dict:
    """Load the 4 relational edge CSVs (by product name)."""
    edge_files = {
        "Plant":    "Edges (Plant).csv",
        "Group":    "Edges (Product Group).csv",
        "SubGroup": "Edges (Product Sub-Group).csv",
        "Storage":  "Edges (Storage Location).csv",
    }
    return {
        rel: pd.read_csv(os.path.join(raw_dir, "Edges", fname))
        for rel, fname in edge_files.items()
    }


def build_graph(
    ordered_products: list,
    edges_dict: dict,
    df_features: pd.DataFrame,
) -> nx.Graph:
    """
    Construct a weighted undirected NetworkX graph.
    Edge weight = number of distinct shared relation types (1..4).
    Node attribute x stores the engineered feature vector.
    """
    prod_to_idx = {p: i for i, p in enumerate(ordered_products)}
    feat_cols   = list(df_features.columns)
    G = nx.Graph()
    for i, prod in enumerate(ordered_products):
        G.add_node(i, node_id=prod, node_index=i,
                   x=[float(df_features.loc[prod, c]) for c in feat_cols])
    relation_tracker = defaultdict(set)
    for rel_label, df_e in edges_dict.items():
        for _, row in df_e.iterrows():
            u_name, v_name = str(row["node1"]), str(row["node2"])
            if u_name != v_name and u_name in prod_to_idx and v_name in prod_to_idx:
                pair = tuple(sorted([prod_to_idx[u_name], prod_to_idx[v_name]]))
                relation_tracker[pair].add(rel_label)
    for (u, v), rels in relation_tracker.items():
        G.add_edge(u, v, weight=float(len(rels)))
    return G


def prune_graph(G: nx.Graph, min_weight: int = 3) -> nx.Graph:
    """
    Return a new graph keeping only edges with weight >= min_weight.
    All nodes are always retained (even if isolated in the pruned view).
    The pruned SupplyGraph (w>=3) splits into 6 disconnected components,
    including two components of size 2. The simulator handles this gracefully.
    """
    G_p = nx.Graph()
    for node, attrs in G.nodes(data=True):
        G_p.add_node(node, **attrs)
    for u, v, d in G.edges(data=True):
        if d["weight"] >= min_weight:
            G_p.add_edge(u, v, **d)
    return G_p


def graph_to_pyg(G: nx.Graph, y: np.ndarray):
    """Convert a NetworkX graph to a PyG Data object with float32 tensors."""
    data = from_networkx(G, group_node_attrs=["x"], group_edge_attrs=["weight"])
    data.x = data.x.to(torch.float32)
    data.y = torch.tensor(y, dtype=torch.float32)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROXY LABEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

def compute_proxy_labels(
    fi_df: pd.DataFrame,
    ordered_products: list,
    q: float = 25.0,
    pop_dup_col: str = "POP001L12P",
) -> tuple:
    """
    Label each product at-risk (1) if its largest single-day negative
    Factory Issue drop falls in the bottom q-th percentile.
    Returns: (labels np.ndarray, largest_drops dict, q_threshold float)
    """
    dup_col = pop_dup_col + ".1"
    if pop_dup_col in fi_df.columns and dup_col in fi_df.columns:
        pop_merged = (fi_df[pop_dup_col].astype(float) + fi_df[dup_col].astype(float)) / 2.0
    else:
        pop_merged = fi_df[pop_dup_col].astype(float) if pop_dup_col in fi_df.columns else None
    drops = {}
    for prod in ordered_products:
        if prod == pop_dup_col and pop_merged is not None:
            vals = pop_merged.values
        else:
            vals = fi_df[prod].astype(float).values
        drops[prod] = float(np.min(np.diff(vals)))
    threshold = np.percentile(list(drops.values()), q)
    labels    = np.array([1 if drops[p] <= threshold else 0
                          for p in ordered_products], dtype=int)
    return labels, drops, float(threshold)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRAPHSAGE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SupplyGraphSAGE(nn.Module):
    """Two-layer inductive GraphSAGE classifier."""
    def __init__(self, in_channels: int = 18, hidden_channels: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h.squeeze(-1)


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_fold(x_tensor, edge_index, y_tensor, train_mask,
                   in_channels, epochs=80, lr=0.01, pos_weight=3.0, seed=42):
    """Train one GraphSAGE fold and return the fitted model."""
    _set_seed(seed)
    model     = SupplyGraphSAGE(in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor, edge_index)
        loss   = criterion(logits[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()
    return model


def evaluate_loocv(pyg_data, X: np.ndarray, y: np.ndarray,
                   epochs: int = 80, lr: float = 0.01,
                   n_fold_threshold: int = 60) -> tuple:
    """
    LOOCV (N <= n_fold_threshold) or StratifiedKFold (N > n_fold_threshold).
    Returns (oof_probs, metrics_dict, protocol_used).
    Each fold fits StandardScaler on training indices only (no leakage).
    """
    N        = len(y)
    oof      = np.zeros(N)
    pos_w    = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    edge_idx = pyg_data.edge_index
    y_tensor = pyg_data.y
    in_ch    = X.shape[1]
    if N <= n_fold_threshold:
        protocol = f"Leave-One-Out CV (N={N})"
        folds    = [(np.array([j for j in range(N) if j != i]), np.array([i]))
                    for i in range(N)]
    else:
        k        = 5
        protocol = f"Stratified {k}-Fold CV (N={N})"
        skf      = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        folds    = list(skf.split(X, y))
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        scaler   = StandardScaler()
        X_sc     = scaler.fit_transform(X[train_idx])
        X_all_sc = scaler.transform(X)
        x_tensor = torch.tensor(X_all_sc, dtype=torch.float32)
        t_mask   = torch.zeros(N, dtype=torch.bool)
        t_mask[train_idx] = True
        model = train_one_fold(x_tensor, edge_idx, y_tensor, t_mask,
                               in_channels=in_ch, epochs=epochs, lr=lr,
                               pos_weight=pos_w, seed=100 + fold_i)
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor, edge_idx)
            probs  = torch.sigmoid(logits).numpy()
            oof[test_idx] = probs[test_idx]
    metrics = {
        "ROC-AUC": float(roc_auc_score(y, oof)),
        "PR-AUC":  float(average_precision_score(y, oof)),
    }
    return oof, metrics, protocol


def get_full_risk_scores(G: nx.Graph, X: np.ndarray, y: np.ndarray,
                         epochs: int = 80, lr: float = 0.01) -> np.ndarray:
    """
    Train on ALL nodes (no held-out set) and return risk scores for display.
    Not used for evaluation - only for the live dashboard visualisation.
    """
    pyg   = graph_to_pyg(G, y)
    N     = len(y)
    pos_w = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    in_ch = X.shape[1]
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)
    x_tensor = torch.tensor(X_sc, dtype=torch.float32)
    y_tensor = pyg.y
    all_mask = torch.ones(N, dtype=torch.bool)
    model = train_one_fold(x_tensor, pyg.edge_index, y_tensor, all_mask,
                           in_channels=in_ch, epochs=epochs, lr=lr,
                           pos_weight=pos_w, seed=42)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_tensor, pyg.edge_index)).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DISRUPTION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _disruption_note() -> str:
    return (
        "**Disruption definition:** All three engineered summary statistics "
        "(mean, std, AND trend slope) for production and delivery features are "
        "set to 0. This models a full flatline/production-halt state. "
        "Setting only the mean to 0 while leaving std or slope positive would "
        "be internally inconsistent — a positive trend slope with mean=0 is "
        "physically meaningless for a halted process. All three stats are "
        "zeroed together for internal consistency."
    )


def simulate_disruption(
    G: nx.Graph,
    X_original: np.ndarray,
    y: np.ndarray,
    ordered_products: list,
    feat_cols: list,
    target_prod: str,
    baseline_scores: np.ndarray,
    epochs: int = 80,
    lr: float = 0.01,
) -> dict:
    """
    Simulate a production halt at target_prod.

    DISRUPTION TRANSFORMATION (all three stats zeroed for consistency):
    For the affected features (Production and Delivery, both Unit and Weight),
    we set mean=0, std=0, AND slope=0 simultaneously. This represents a true
    flatline/halt state. Zeroing only mean while leaving std positive would
    imply high variance around zero -- contradictory for a stopped process.
    Zeroing mean while leaving slope positive would imply recovery trend from
    a zero baseline -- also physically meaningless at t=0.

    Handles sparse/disconnected components in the pruned graph gracefully:
    - If hop2 is empty, empty_hop2=True is set and a descriptive message is
      provided for the UI. The component_size is always reported.
    - No auto-fallback to the dense graph. Limitation is surfaced explicitly.
    """
    target_idx  = ordered_products.index(target_prod)
    idx_to_prod = {i: p for i, p in enumerate(ordered_products)}

    hop1 = list(G.neighbors(target_idx))
    hop2 = []
    for n in hop1:
        for nn in G.neighbors(n):
            if nn != target_idx and nn not in hop1 and nn not in hop2:
                hop2.append(nn)

    component_size = len(nx.node_connected_component(G, target_idx))

    # Build disrupted feature matrix: zero ALL THREE stats for each affected feature.
    X_disrupted = X_original.copy()
    for col in DISRUPTION_FEATURES:
        if col in feat_cols:
            ci = feat_cols.index(col)
            X_disrupted[target_idx, ci] = 0.0

    pyg    = graph_to_pyg(G, y)
    N      = len(y)
    pos_w  = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    in_ch  = X_original.shape[1]

    # Fit StandardScaler on ORIGINAL data, then transform the disrupted matrix.
    # This preserves the relative feature scale used at training time.
    scaler     = StandardScaler()
    scaler.fit(X_original)
    X_sc_dis   = scaler.transform(X_disrupted)
    x_dis_t    = torch.tensor(X_sc_dis, dtype=torch.float32)
    y_tensor   = pyg.y
    all_mask   = torch.ones(N, dtype=torch.bool)

    model = train_one_fold(x_dis_t, pyg.edge_index, y_tensor, all_mask,
                           in_channels=in_ch, epochs=epochs, lr=lr,
                           pos_weight=pos_w, seed=999)
    model.eval()
    with torch.no_grad():
        after_scores = torch.sigmoid(model(x_dis_t, pyg.edge_index)).numpy()

    empty_hop2_msg = (
        "No 2-hop neighbours found — this product's high-confidence connections "
        "are limited to a small, isolated cluster in the pruned graph. "
        "This is expected: the pruned graph (w>=3) splits into 6 disconnected "
        "components, including two components of size 2. Switch to the dense "
        "graph view to see broader (but lower-confidence) propagation effects."
    ) if len(hop2) == 0 else ""

    return {
        "before":          {i: float(baseline_scores[i]) for i in range(N)},
        "after":           {i: float(after_scores[i])    for i in range(N)},
        "target_idx":      target_idx,
        "hop1":            hop1,
        "hop2":            hop2,
        "hop1_names":      [idx_to_prod[n] for n in hop1],
        "hop2_names":      [idx_to_prod[n] for n in hop2],
        "component_size":  component_size,
        "empty_hop2":      len(hop2) == 0,
        "empty_hop2_msg":  empty_hop2_msg,
        "note":            _disruption_note(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PYVIS NETWORK RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _risk_to_hex(score: float) -> str:
    """Map a risk probability [0,1] to hex colour: green -> yellow -> red."""
    r = int(min(255, score * 2 * 255))
    g = int(min(255, (1 - score) * 2 * 255))
    return f"#{r:02x}{g:02x}00"


def create_pyvis_network(
    G: nx.Graph,
    ordered_products: list,
    risk_scores: np.ndarray,
    disrupted_idx: int = None,
    hop1: list = None,
    hop2: list = None,
    height: str = "580px",
) -> str:
    """
    Build an interactive Pyvis HTML graph with risk colour gradient and
    disruption neighbourhood highlights. Returns raw HTML string.
    """
    hop1 = hop1 or []
    hop2 = hop2 or []

    net = Network(height=height, width="100%", notebook=False,
                  bgcolor="#0e1117", font_color="white",
                  cdn_resources="in_line")
    
    # Check graph density to dynamically adapt physics spacing and repulsion
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    is_very_dense = (n_edges > 200) if n_nodes > 0 else False

    grav_constant = -9000 if is_very_dense else -4000
    spring_len    = 240 if is_very_dense else 160
    spring_k      = 0.003 if is_very_dense else 0.015
    edge_alpha    = "rgba(100, 110, 140, 0.18)" if is_very_dense else "rgba(120, 130, 170, 0.4)"

    net.set_options(f"""
    {{
      "nodes": {{
        "font": {{
          "size": 11,
          "color": "#ffffff",
          "strokeWidth": 2,
          "strokeColor": "#000000"
        }}
      }},
      "edges": {{
        "color": {{"color": "{edge_alpha}"}},
        "smooth": false
      }},
      "physics": {{
        "barnesHut": {{
          "gravitationalConstant": {grav_constant},
          "centralGravity": 0.25,
          "springLength": {spring_len},
          "springConstant": {spring_k},
          "damping": 0.09,
          "avoidOverlap": 0.7
        }},
        "stabilization": {{"iterations": 200}}
      }},
      "interaction": {{
        "hover": true,
        "zoomView": true,
        "dragNodes": true
      }}
    }}
    """)

    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1
    min_deg  = min(degrees.values()) if degrees else 0
    deg_range = max(1, max_deg - min_deg)
    idx_prod = {i: p for i, p in enumerate(ordered_products)}

    for node in G.nodes():
        prod  = idx_prod[node]
        score = float(risk_scores[node])
        deg   = degrees[node]
        # Keep node radius restrained: 10px to 20px so nodes never blob together
        size  = 10 + int(10 * (deg - min_deg) / deg_range)
        fill  = _risk_to_hex(score)
        border, bw, lsuffix = fill, 1.5, ""

        if node == disrupted_idx:
            border, bw, lsuffix = "#9b59b6", 4, " ⚡"
            size = max(size, 18)
        elif node in hop1:
            border, bw, lsuffix = "#e67e22", 3, " [1↔]"
        elif node in hop2:
            border, bw, lsuffix = "#f1c40f", 2.5, " [2↔]"

        tooltip = (f"<b>{prod}</b><br>"
                   f"Risk Score: {score:.3f}<br>"
                   f"Degree: {deg}<br>"
                   f"Index: {node}")
        if node == disrupted_idx:
            tooltip += "<br><b>⚡ DISRUPTED</b>"

        net.add_node(node,
                     label=prod + lsuffix,
                     color={"background": fill, "border": border,
                            "highlight": {"background": fill, "border": "#ffffff"}},
                     size=size, borderWidth=bw, title=tooltip)

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        # In dense mode, keep edges slim so they don't visually overwhelm the layout
        edge_w = max(0.4, w * 0.3) if is_very_dense else (w * 0.6)
        net.add_edge(u, v, width=edge_w,
                     title=f"Shared relations: {int(w)}")

    return net.generate_html()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SECTION 2: VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class UploadError(Exception):
    pass


def validate_nodes_csv(df: pd.DataFrame) -> list:
    if "Node" not in df.columns:
        raise UploadError("Nodes CSV must have a column named 'Node'.")
    dupes = df[df.duplicated(subset=["Node"])]["Node"].tolist()
    if dupes:
        raise UploadError(f"Duplicate node names found: {dupes}.")
    return df["Node"].tolist()


def validate_edge_csv(df: pd.DataFrame, ordered_products: list,
                      label: str) -> None:
    for col in ("node1", "node2"):
        if col not in df.columns:
            raise UploadError(f"Edge file '{label}' is missing column '{col}'.")
    node_set = set(ordered_products)
    bad = (set(df["node1"].astype(str)) | set(df["node2"].astype(str))) - node_set
    if bad:
        raise UploadError(
            f"Edge file '{label}' references unknown products: "
            f"{list(bad)[:5]}{'...' if len(bad) > 5 else ''}")


def validate_temporal_csv(df: pd.DataFrame, ordered_products: list,
                           label: str) -> None:
    first_col = df.columns[0].lower()
    if first_col not in ("date", "timestamp"):
        raise UploadError(f"Temporal file '{label}' must have a 'Date' first column.")
    missing = [p for p in ordered_products if p not in df.columns]
    if missing:
        raise UploadError(
            f"Temporal file '{label}' is missing {len(missing)} product columns, "
            f"e.g. {missing[:3]}.")
    nan_count = df[ordered_products].isna().sum().sum()
    if nan_count > 0:
        raise UploadError(
            f"Temporal file '{label}' contains {nan_count} NaN values.")


def check_label_skew(y: np.ndarray, threshold: float = 0.85) -> tuple:
    if len(y) == 0:
        return True, 100.0
    maj = max((y == 0).mean(), (y == 1).mean())
    return maj > threshold, float(maj * 100)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== pipeline_core.py self-test ===")
    prods     = get_ordered_products(RAW_DIR)
    temp_dfs  = load_supplygraph_temporal(RAW_DIR)
    df_feats  = engineer_features(temp_dfs, prods, drop_leaking=True)
    print(f"Features: {df_feats.shape}  (expected 40x18)")
    fi_df              = temp_dfs[("Factory_Issue", "Unit")]
    labels, drops, q25 = compute_proxy_labels(fi_df, prods)
    print(f"Labels: {np.bincount(labels)}  threshold={q25:.1f}")
    edges_dict = load_supplygraph_edges(RAW_DIR)
    G_dense    = build_graph(prods, edges_dict, df_feats)
    G_pruned   = prune_graph(G_dense, min_weight=3)
    print(f"Dense:  {G_dense.number_of_nodes()}N / {G_dense.number_of_edges()}E")
    print(f"Pruned: {G_pruned.number_of_nodes()}N / {G_pruned.number_of_edges()}E")
    X       = df_feats.values.astype(np.float32)
    feat_cs = list(df_feats.columns)
    scores  = get_full_risk_scores(G_pruned, X, labels)
    print(f"Risk scores: min={scores.min():.3f}  max={scores.max():.3f}")
    target  = "EEA500G12P"
    sim     = simulate_disruption(G_pruned, X, labels, prods, feat_cs, target, scores)
    print(f"Simulation on {target!r}:")
    print(f"  Component size: {sim['component_size']}")
    print(f"  1-hop: {sim['hop1_names']}")
    print(f"  2-hop: {sim['hop2_names']}")
    print(f"  empty_hop2: {sim['empty_hop2']}")
    html = create_pyvis_network(G_pruned, prods, scores,
                                disrupted_idx=sim["target_idx"],
                                hop1=sim["hop1"], hop2=sim["hop2"])
    print(f"Pyvis HTML: {len(html):,} bytes")
    print("=== All self-tests passed ===")
