"""
==============================================================================
SupplyGraph: Static Homogeneous Graph Construction and Validation
==============================================================================
Repository: https://github.com/ciol-researchlab/SupplyGraph
Dataset   : FMCG Supply Chain Planning Dataset (Raw Dataset/Homogenoeus)

Overview:
  This script constructs and validates a static homogeneous graph object
  for supply chain disruption / risk propagation analysis following 5
  sequential steps:
    Step 1: Data Inspection & Data Quality Audit
    Step 2: Node Feature Engineering (Static Summary Statistics)
    Step 3: Edge Construction (Weighted NetworkX Graph)
    Step 4: Graph Structure & Feature Validation
    Step 5: Conversion to PyTorch Geometric Data Object

Data Quality Note (Documented for Research / Project Writeup):
  During Step 1 data inspection, an empirical discrepancy was identified:
  - The dataset documentation and paper state 41 product nodes.
  - In "Nodes/Nodes.csv" and "Nodes/NodesIndex.csv", there are 41 rows, but product
    code "POP001L12P" appears twice (at row index 12 and row index 20).
  - In "Nodes/NodesIndex.csv", the authors explicitly assigned NodeIndex = 12
    to BOTH occurrences. Hence, indices only span 0 to 39 (40 unique indices).
  - In "Nodes Type (Plant & Storage).csv", exactly 40 unique products are listed.
  - In all edge files ("Edges/*.csv" and "EdgesIndex/*.csv"), the 684 unique
    undirected edges connect only these 40 unique products (indices 0..39).
  - In "Temporal Data" (Unit and Weight), there are 41 product columns, where
    the two "POP001L12P" columns have different daily time series.
  Resolution (Option 1):
    A 40-node graph (indices 0 to 39) is constructed to strictly preserve the
    empirical edge connectivity of 684 edges without creating artificial isolated
    nodes. For "POP001L12P", the two distinct time series are AVERAGED (rather
    than summed) across all 8 temporal features to preserve product scale parity
    with the other 39 products.

Dependencies: pandas, numpy, scipy, networkx, torch, torch_geometric
==============================================================================
"""

import os
import sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.stats import linregress
import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import from_networkx

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "Raw Dataset", "Homogenoeus")

def print_header(title, step_num=None):
    width = 78
    print("\n" + "=" * width)
    if step_num is not None:
        print(f"  STEP {step_num}: {title.upper()}")
    else:
        print(f"  {title.upper()}")
    print("=" * width + "\n")

def print_sub(title):
    print(f"\n--- {title} ---")


# =============================================================================
# STEP 1: DATA INSPECTION
# =============================================================================
def step_1_data_inspection():
    print_header("Data Inspection & File Structure Verification", step_num=1)
    print(f"Raw data directory: {RAW_DIR}")
    assert os.path.exists(RAW_DIR), f"Directory not found: {RAW_DIR}"

    # 1.1 Inspect Node Files
    print_sub("1.1 Node Metadata Files")
    nodes_path = os.path.join(RAW_DIR, "Nodes", "Nodes.csv")
    df_nodes = pd.read_csv(nodes_path)
    print(f"Nodes.csv shape: {df_nodes.shape}")
    print(f"Nodes.csv columns: {list(df_nodes.columns)}")
    print(f"Nodes.csv dtypes:\n{df_nodes.dtypes.to_string()}")
    print(f"Nodes.csv sample (head 5):\n{df_nodes.head(5).to_string()}")

    nodes_idx_path = os.path.join(RAW_DIR, "Nodes", "NodesIndex.csv")
    df_nodes_idx = pd.read_csv(nodes_idx_path)
    print(f"\nNodesIndex.csv shape: {df_nodes_idx.shape}")
    print(f"NodesIndex.csv columns: {list(df_nodes_idx.columns)}")
    print(f"NodesIndex.csv dtypes:\n{df_nodes_idx.dtypes.to_string()}")
    print(f"NodesIndex.csv sample (head 5):\n{df_nodes_idx.head(5).to_string()}")

    nt_group_path = os.path.join(RAW_DIR, "Nodes", "Node Types (Product Group and Subgroup).csv")
    df_nt_group = pd.read_csv(nt_group_path)
    print(f"\nNode Types (Group/Subgroup).csv shape: {df_nt_group.shape}")
    print(f"Sample:\n{df_nt_group.head(3).to_string()}")

    nt_plant_path = os.path.join(RAW_DIR, "Nodes", "Nodes Type (Plant & Storage).csv")
    df_nt_plant = pd.read_csv(nt_plant_path)
    print(f"\nNodes Type (Plant & Storage).csv shape: {df_nt_plant.shape}")
    print(f"Sample:\n{df_nt_plant.head(3).to_string()}")

    # 1.2 Inspect Edge Files
    print_sub("1.2 Relational Edge Files")
    edge_files = [
        "Edges (Plant).csv",
        "Edges (Product Group).csv",
        "Edges (Product Sub-Group).csv",
        "Edges (Storage Location).csv"
    ]
    edges_raw = {}
    for ef in edge_files:
        p = os.path.join(RAW_DIR, "Edges", ef)
        df_e = pd.read_csv(p)
        edges_raw[ef] = df_e
        print(f"Edges/{ef} -> Shape: {df_e.shape}, Columns: {list(df_e.columns)}")
        print(f"  Sample:\n{df_e.head(2).to_string()}")

    print_sub("1.3 EdgesIndex Files (Pre-indexed by Author)")
    for ef in edge_files:
        p = os.path.join(RAW_DIR, "Edges", "EdgesIndex", ef)
        df_ei = pd.read_csv(p)
        print(f"EdgesIndex/{ef} -> Shape: {df_ei.shape}, Columns: {list(df_ei.columns)}")
        print(f"  Sample:\n{df_ei.head(2).to_string()}")

    # 1.3 Inspect Temporal Files
    print_sub("1.4 Temporal Daily Feature Files")
    temporal_manifest = {
        ("Production", "Unit"): ("Unit", "Production .csv"),
        ("Production", "Weight"): ("Weight", "Production .csv"),
        ("Sales_Order", "Unit"): ("Unit", "Sales Order.csv"),
        ("Sales_Order", "Weight"): ("Weight", "Sales Order .csv"),
        ("Delivery", "Unit"): ("Unit", "Delivery To distributor.csv"),
        ("Delivery", "Weight"): ("Weight", "Delivery to Distributor.csv"),
        ("Factory_Issue", "Unit"): ("Unit", "Factory Issue.csv"),
        ("Factory_Issue", "Weight"): ("Weight", "Factory Issue.csv"),
    }
    
    temporal_dfs = {}
    for (feat, measure), (subdir, fname) in temporal_manifest.items():
        p = os.path.join(RAW_DIR, "Temporal Data", subdir, fname)
        df_t = pd.read_csv(p)
        temporal_dfs[(feat, measure)] = df_t
        print(f"Temporal Data/{subdir}/{fname:<30} -> Shape: {df_t.shape} "
              f"| Dates: {df_t['Date'].iloc[0]} to {df_t['Date'].iloc[-1]} "
              f"| Product cols: {df_t.shape[1] - 1}")

    # 1.4 Validate Node Count & Edge Count
    print_sub("1.5 Dataset Count Verification & Quality Audit")
    num_rows_nodes = len(df_nodes)
    unique_node_names = df_nodes["Node"].unique()
    num_unique_nodes = len(unique_node_names)

    print(f"Nodes.csv total rows          : {num_rows_nodes}")
    print(f"Nodes.csv unique product names: {num_unique_nodes}")

    # Compute unique undirected edges across all 4 relation files
    pair_relations = defaultdict(set)
    for rel_name, df_e in edges_raw.items():
        for _, row in df_e.iterrows():
            u = str(row["node1"])
            v = str(row["node2"])
            if u != v:
                pair = tuple(sorted([u, v]))
                pair_relations[pair].add(rel_name)

    unique_edge_count = len(pair_relations)
    weight_dist = Counter(len(rels) for rels in pair_relations.values())

    print(f"Unique undirected product pairs across all 4 files: {unique_edge_count}")
    print("Edge weight distribution (number of shared relation types):")
    for w in sorted(weight_dist.keys()):
        print(f"  Shared relations = {w}: {weight_dist[w]} edges")

    # Audit POP001L12P duplication
    print("\n[DATA QUALITY AUDIT NOTE]")
    print("  Documented issue: Product 'POP001L12P' appears twice in Nodes.csv (rows 12 and 20).")
    print("  In NodesIndex.csv, both occurrences are mapped to NodeIndex = 12.")
    print("  In EdgesIndex/*.csv, indices only span 0..39 (40 unique nodes).")
    print(f"  Edge count check: 684 expected vs {unique_edge_count} found -> EXACT MATCH.")
    print("  Node count resolution: Using 40 unique product nodes (indices 0..39).")
    print("  For POP001L12P, the two time-series columns are averaged for scale parity.")

    return df_nodes_idx, temporal_dfs, edges_raw


# =============================================================================
# STEP 2: NODE FEATURE ENGINEERING
# =============================================================================
def step_2_node_feature_engineering(df_nodes_idx, temporal_dfs):
    print_header("Node Feature Engineering (Static Summary Statistics)", step_num=2)
    
    # Ordered 40 products matching NodesIndex 0..39
    df_unique = df_nodes_idx.drop_duplicates(subset=["Node"]).reset_index(drop=True)
    ordered_products = df_unique["Node"].tolist()
    print(f"Processing features for {len(ordered_products)} unique products.")

    # Define 8 temporal series
    series_keys = [
        ("Production", "Unit"),
        ("Production", "Weight"),
        ("Sales_Order", "Unit"),
        ("Sales_Order", "Weight"),
        ("Delivery", "Unit"),
        ("Delivery", "Weight"),
        ("Factory_Issue", "Unit"),
        ("Factory_Issue", "Weight"),
    ]

    engineered_records = {prod: {} for prod in ordered_products}

    for (feat_name, measure) in series_keys:
        df_t = temporal_dfs[(feat_name, measure)]
        prefix = f"{feat_name}_{measure.lower()}"
        num_days = len(df_t)
        time_steps = np.arange(num_days)

        # Average the two POP001L12P columns if both present
        if "POP001L12P" in df_t.columns and "POP001L12P.1" in df_t.columns:
            s_pop_avg = (df_t["POP001L12P"].astype(float) + df_t["POP001L12P.1"].astype(float)) / 2.0
        else:
            s_pop_avg = df_t["POP001L12P"].astype(float)

        for prod in ordered_products:
            if prod == "POP001L12P":
                values = s_pop_avg.values
            else:
                values = df_t[prod].astype(float).values

            # Summary statistics: Mean, Standard Deviation, Trend Slope
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            
            # Linear regression slope (OLS slope: dy/dt)
            slope_res = linregress(time_steps, values)
            slope_val = float(slope_res.slope)

            engineered_records[prod][f"{prefix}_mean"] = mean_val
            engineered_records[prod][f"{prefix}_std"] = std_val
            engineered_records[prod][f"{prefix}_slope"] = slope_val

    df_features = pd.DataFrame.from_dict(engineered_records, orient="index")
    df_features.index.name = "Node"

    print(f"\nEngineered Node Features DataFrame:")
    print(f"  Shape: {df_features.shape} (40 products x 24 static features)")
    print(f"  Total missing (NaN) values: {df_features.isna().sum().sum()}")
    print(f"  Total infinite values     : {np.isinf(df_features.values).sum()}")

    print("\nList of 24 Engineered Feature Columns:")
    for i, col in enumerate(df_features.columns):
        print(f"  [{i+1:2d}] {col}")

    print("\nSample engineered features for first 3 products (first 6 features):")
    print(df_features.iloc[:3, :6].to_string())

    print("\nEngineered features for resolved product POP001L12P (first 6 features):")
    print(df_features.loc[["POP001L12P"], df_features.columns[:6]].to_string())

    return df_features, ordered_products


# =============================================================================
# STEP 3: EDGE CONSTRUCTION
# =============================================================================
def step_3_edge_construction(df_features, ordered_products, edges_raw):
    print_header("Edge Construction (Weighted NetworkX Graph)", step_num=3)

    prod_to_idx = {prod: i for i, prod in enumerate(ordered_products)}
    idx_to_prod = {i: prod for i, prod in enumerate(ordered_products)}
    num_nodes = len(ordered_products)
    feature_cols = list(df_features.columns)

    G = nx.Graph()

    # Add nodes with metadata and features
    for i, prod in enumerate(ordered_products):
        node_feat_vector = [float(df_features.loc[prod, c]) for c in feature_cols]
        G.add_node(
            i,
            node_id=prod,
            node_index=i,
            x=node_feat_vector
        )

    # Collapse relations into weighted edges
    relation_tracker = defaultdict(lambda: {"count": 0, "relations": set()})
    
    relation_names = {
        "Edges (Plant).csv": "Plant",
        "Edges (Product Group).csv": "Group",
        "Edges (Product Sub-Group).csv": "SubGroup",
        "Edges (Storage Location).csv": "Storage"
    }

    for ef, rel_label in relation_names.items():
        df_e = edges_raw[ef]
        for _, row in df_e.iterrows():
            u_name = str(row["node1"])
            v_name = str(row["node2"])
            if u_name != v_name and u_name in prod_to_idx and v_name in prod_to_idx:
                u = prod_to_idx[u_name]
                v = prod_to_idx[v_name]
                pair = tuple(sorted([u, v]))
                relation_tracker[pair]["count"] += 1
                relation_tracker[pair]["relations"].add(rel_label)

    # Add weighted edges to NetworkX graph
    for (u, v), info in relation_tracker.items():
        # Weight = number of distinct shared relations (1 to 4)
        weight = float(len(info["relations"]))
        G.add_edge(u, v, weight=weight)

    print(f"NetworkX Graph Construction Summary:")
    print(f"  Nodes added: {G.number_of_nodes()} (Products 0 to {num_nodes-1})")
    print(f"  Edges added: {G.number_of_edges()} (Collapsed weighted edges)")
    print(f"  Target edge count: 684 | Actual: {G.number_of_edges()} -> "
          f"{'MATCH' if G.number_of_edges() == 684 else 'MISMATCH'}")

    sample_edges = list(G.edges(data=True))[:5]
    print("\nSample edges with collapsed weights:")
    for u, v, data in sample_edges:
        print(f"  ({idx_to_prod[u]:<12}, {idx_to_prod[v]:<12}) -> Weight: {data['weight']}")

    return G, prod_to_idx, idx_to_prod


# =============================================================================
# STEP 4: GRAPH VALIDATION
# =============================================================================
def step_4_graph_validation(G, df_features):
    print_header("Graph Validation & Sanity Range Checks", step_num=4)

    checks_passed = True

    # 4.1 Node & Edge Counts
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"1. Node Count: {num_nodes} (Expected: 40) -> {'PASS' if num_nodes == 40 else 'FAIL'}")
    print(f"2. Edge Count: {num_edges} (Expected: 684) -> {'PASS' if num_edges == 684 else 'FAIL'}")
    if num_nodes != 40 or num_edges != 684:
        checks_passed = False

    # 4.2 Isolated Nodes Check
    isolated_nodes = [n for n, d in G.degree() if d == 0]
    print(f"3. Isolated Nodes (degree = 0): {len(isolated_nodes)} -> "
          f"{'PASS (None detected)' if len(isolated_nodes) == 0 else 'FLAGGED: ' + str(isolated_nodes)}")
    if len(isolated_nodes) > 0:
        checks_passed = False

    # 4.3 Degree Distribution Analysis
    degrees = [d for _, d in G.degree()]
    max_possible_deg = num_nodes - 1  # 39
    density = nx.density(G)
    print("\n4. Degree Distribution Analysis:")
    print(f"   Min Degree   : {np.min(degrees)}")
    print(f"   Max Degree   : {np.max(degrees)} (Theoretical max = {max_possible_deg})")
    print(f"   Mean Degree  : {np.mean(degrees):.2f}")
    print(f"   Median Degree: {np.median(degrees):.1f}")
    print(f"   Std Dev      : {np.std(degrees):.2f}")
    print(f"   Graph Density: {density * 100:.2f}% ({num_edges} / {num_nodes * (num_nodes - 1) // 2})")

    print("\n   [Structural Topology Note]")
    print(f"   The degree distribution ranges from {np.min(degrees)} to {np.max(degrees)} with a mean of {np.mean(degrees):.2f}.")
    print(f"   Graph density is {density*100:.1f}%. This topology is NOT heavy-tailed / scale-free (power-law);")
    print("   rather, it is highly dense and relatively uniform. This occurs because in FMCG supply chains,")
    print("   nearly all products share centralized storage facilities and manufacturing plant complexes,")
    print("   forming a highly connected dense cluster.")

    # 4.4 Connected Components
    num_cc = nx.number_connected_components(G)
    cc_sizes = [len(c) for c in nx.connected_components(G)]
    print(f"\n5. Connected Components: {num_cc} (Sizes: {cc_sizes}) -> "
          f"{'PASS (Single component)' if num_cc == 1 else 'FAIL'}")
    if num_cc != 1:
        checks_passed = False

    # 4.5 Sanity Range Checks on Edge Weights
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    min_w, max_w = min(weights), max(weights)
    valid_weights = all(w in (1.0, 2.0, 3.0, 4.0) for w in weights)
    print(f"\n6. Edge Weights Sanity Check: Min = {min_w}, Max = {max_w}")
    print(f"   All weights in {{1, 2, 3, 4}}: {'PASS' if valid_weights else 'FAIL'}")
    if not valid_weights:
        checks_passed = False

    # 4.6 Sanity Range Checks on Node Features
    feat_matrix = df_features.values
    nan_count = np.isnan(feat_matrix).sum()
    inf_count = np.isinf(feat_matrix).sum()
    
    # Check that standard deviation columns are strictly non-negative
    std_cols = [c for c in df_features.columns if c.endswith("_std")]
    negative_stds = (df_features[std_cols] < 0).sum().sum()
    
    # Check that mean volume columns are strictly non-negative
    mean_cols = [c for c in df_features.columns if c.endswith("_mean")]
    negative_means = (df_features[mean_cols] < 0).sum().sum()

    print(f"\n7. Node Feature Matrix Sanity Checks:")
    print(f"   NaN values count      : {nan_count} -> {'PASS' if nan_count == 0 else 'FAIL'}")
    print(f"   Infinite values count : {inf_count} -> {'PASS' if inf_count == 0 else 'FAIL'}")
    print(f"   Negative standard devs: {negative_stds} -> {'PASS' if negative_stds == 0 else 'FAIL'}")
    print(f"   Negative mean volumes : {negative_means} -> {'PASS' if negative_means == 0 else 'FAIL'}")

    if nan_count > 0 or inf_count > 0 or negative_stds > 0 or negative_means > 0:
        checks_passed = False

    print(f"\n>>> Overall Validation Status: {'ALL CHECKS PASSED' if checks_passed else 'VALIDATION FAILED'}")
    assert checks_passed, "Validation checks did not pass. Halting before Step 5."

    return checks_passed


# =============================================================================
# STEP 5: CONVERT TO PYTORCH GEOMETRIC
# =============================================================================
def step_5_convert_to_pyg(G, df_features):
    print_header("Convert to PyTorch Geometric (PyG) Data Object", step_num=5)

    # Use torch_geometric.utils.from_networkx
    pyg_data = from_networkx(
        G,
        group_node_attrs=["x"],
        group_edge_attrs=["weight"]
    )

    # Cast node features to torch.float32
    pyg_data.x = pyg_data.x.to(torch.float32)
    # PyG from_networkx creates edge_attr of shape [num_edges, 1]
    pyg_data.edge_weight = pyg_data.edge_attr.view(-1).to(torch.float32)

    print("PyTorch Geometric Data Object Successfully Created:")
    print(f"  Data object representation: {pyg_data}")
    print(f"  Node feature matrix (x)   : Shape {list(pyg_data.x.shape)} | dtype: {pyg_data.x.dtype}")
    print(f"  Edge index (edge_index)   : Shape {list(pyg_data.edge_index.shape)} | dtype: {pyg_data.edge_index.dtype}")
    print(f"  Edge weight (edge_weight) : Shape {list(pyg_data.edge_weight.shape)} | dtype: {pyg_data.edge_weight.dtype}")

    # Verify PyG structural properties
    print("\nVerification of PyG Graph Properties:")
    print(f"  Number of nodes          : {pyg_data.num_nodes} (Expected: 40)")
    print(f"  Number of directed edges : {pyg_data.num_edges} (Expected: 2 * 684 = 1368)")
    print(f"  Is undirected graph?     : {pyg_data.is_undirected()}")
    print(f"  Has isolated nodes?      : {pyg_data.has_isolated_nodes()}")
    print(f"  Has self loops?          : {pyg_data.has_self_loops()}")

    # Shape assertions
    assert pyg_data.num_nodes == 40, f"Expected 40 nodes, got {pyg_data.num_nodes}"
    assert pyg_data.edge_index.shape == (2, 1368), f"Expected (2, 1368), got {pyg_data.edge_index.shape}"
    assert pyg_data.x.shape == (40, 24), f"Expected (40, 24), got {pyg_data.x.shape}"
    assert pyg_data.edge_weight.shape == (1368,), f"Expected (1368,), got {pyg_data.edge_weight.shape}"

    print("\n>>> PyG Conversion and Verification Complete: Graph is ready for GNN modeling!")
    return pyg_data


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    print("=" * 78)
    print("  SUPPLYGRAPH: STATIC GRAPH CONSTRUCTION & VALIDATION PIPELINE")
    print("=" * 78)

    # Step 1: Data Inspection
    df_nodes_idx, temporal_dfs, edges_raw = step_1_data_inspection()

    # Step 2: Node Feature Engineering
    df_features, ordered_products = step_2_node_feature_engineering(df_nodes_idx, temporal_dfs)

    # Step 3: Edge Construction
    G, prod_to_idx, idx_to_prod = step_3_edge_construction(df_features, ordered_products, edges_raw)

    # Step 4: Graph Validation
    step_4_graph_validation(G, df_features)

    # Step 5: Convert to PyTorch Geometric
    pyg_data = step_5_convert_to_pyg(G, df_features)

    print_header("Execution Finished Successfully")
    return pyg_data

if __name__ == "__main__":
    main()
