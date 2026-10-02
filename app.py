"""
app.py — GraphRisk: Supply Chain Disruption Prediction Engine
=============================================================
Streamlit dashboard with two sections:

  Section 1 (Fixed SupplyGraph Demo)
    - Interactive graph visualizer (dense / pruned toggle)
    - Disruption simulator with 1-hop / 2-hop risk propagation
    - Model inference panel with Phase 1 benchmark metrics

  Section 2 (Bring Your Own Data)
    - Strict SupplyGraph-schema file upload
    - Validation, fresh model retrain, full dashboard replay

Both sections share all core functions from pipeline_core.py.
"""

import io
import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# ── project root on the path ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline_core import (
    RAW_DIR, PHASE1_METRICS, DISRUPTION_FEATURES,
    get_ordered_products, load_supplygraph_temporal,
    engineer_features, compute_proxy_labels,
    load_supplygraph_edges, build_graph, prune_graph,
    graph_to_pyg, get_full_risk_scores, evaluate_loocv,
    simulate_disruption, create_pyvis_network,
    UploadError, validate_nodes_csv, validate_edge_csv,
    validate_temporal_csv, check_label_skew,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRisk",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .proxy-label-box {
      background: #2d1a00; border: 1px solid #e67e22; border-radius: 6px;
      padding: 10px 14px; margin-bottom: 12px;
  }
  .metric-card {
      background: #1a1a2e; border-radius: 8px;
      padding: 14px 18px; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED SUPPLYGRAPH DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SupplyGraph dataset…")
def load_sg_artifacts():
    prods      = get_ordered_products(RAW_DIR)
    temp_dfs   = load_supplygraph_temporal(RAW_DIR)
    df_feats   = engineer_features(temp_dfs, prods, drop_leaking=True)
    fi_df      = temp_dfs[("Factory_Issue", "Unit")]
    labels, _, q25 = compute_proxy_labels(fi_df, prods)
    edges_dict = load_supplygraph_edges(RAW_DIR)
    G_dense    = build_graph(prods, edges_dict, df_feats)
    G_pruned   = prune_graph(G_dense, min_weight=3)
    X          = df_feats.values.astype(np.float32)
    feat_cols  = list(df_feats.columns)
    return prods, df_feats, labels, q25, G_dense, G_pruned, X, feat_cols


@st.cache_data(show_spinner="Computing risk scores (dense graph)…")
def cached_dense_scores(_G, _X, _y):
    import numpy as _np
    return get_full_risk_scores(_G, _X, _y)


@st.cache_data(show_spinner="Computing risk scores (pruned graph)…")
def cached_pruned_scores(_G, _X, _y):
    return get_full_risk_scores(_G, _X, _y)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DASHBOARD COMPONENT: GRAPH + SIMULATOR + METRICS PANEL
# ─────────────────────────────────────────────────────────────────────────────
def render_graph_panel(prods, G_dense, G_pruned, X, labels, feat_cols,
                       dense_scores, pruned_scores,
                       section_key: str):
    """
    Renders the interactive graph visualizer, disruption simulator,
    and model inference panel.  Works for both Section 1 and Section 2
    via the `section_key` namespace for Streamlit widget keys.
    """
    col_left, col_right = st.columns([3, 2])

    # ── Graph topology selector ──────────────────────────────────────────────
    with col_left:
        st.subheader("🔗 Supply-Chain Network")

        graph_choice = st.radio(
            "Graph topology",
            options=["Dense graph (w ≥ 1, 684 edges)",
                     "Pruned graph (w ≥ 3, 136 edges)"],
            key=f"{section_key}_graph_choice",
            horizontal=True,
        )

        is_pruned = graph_choice.startswith("Pruned")
        G_active  = G_pruned if is_pruned else G_dense
        scores    = pruned_scores if is_pruned else dense_scores

        # ── Pruned-graph topology note ────────────────────────────────────
        if is_pruned:
            st.info(
                "**Pruned graph topology note:** Filtering to w ≥ 3 retains "
                "only edges where two products share ≥ 3 operational "
                "categories (plant, storage, group, sub-group). This splits "
                "the 40-node graph into **6 disconnected components** — "
                "including two clusters of size 2. Some products will "
                "therefore have no 2-hop neighbours in the simulator. "
                "This is a meaningful finding from Phase 1 (not a bug): "
                "the dense graph shows broader but noisier propagation, "
                "while the pruned graph shows tighter but sparser connections."
            )

        # Disrupt state
        disrupt_state = st.session_state.get(f"{section_key}_sim_result", None)

        html = create_pyvis_network(
            G_active, prods, scores,
            disrupted_idx=disrupt_state["target_idx"] if disrupt_state else None,
            hop1=disrupt_state["hop1"] if disrupt_state else None,
            hop2=disrupt_state["hop2"] if disrupt_state else None,
        )
        components.html(html, height=610, scrolling=False)

        # Colour legend
        st.markdown(
            "🟢 Low risk &nbsp;&nbsp; 🟡 Medium risk &nbsp;&nbsp; 🔴 High risk &nbsp;&nbsp; "
            "**🟣 Disrupted node** &nbsp;&nbsp; **🟠 1-hop neighbours** &nbsp;&nbsp; "
            "**🟡 2-hop neighbours**",
            unsafe_allow_html=True,
        )

    # ── Disruption Simulator ─────────────────────────────────────────────────
    with col_right:
        st.subheader("⚡ Disruption Simulator")

        st.markdown(
            "<div class='proxy-label-box'>"
            "<b>⚠️ Proxy simulation:</b> Disruption sets <em>mean, std, AND trend "
            "slope</em> to 0 for all production and delivery features — modelling a "
            "full production flatline. Only zeroing the mean (leaving std/slope "
            "non-zero) would be physically inconsistent for a halted process."
            "</div>",
            unsafe_allow_html=True,
        )

        prod_choice = st.selectbox(
            "Select product to disrupt",
            options=prods,
            key=f"{section_key}_prod_choice",
        )

        run_btn = st.button(
            "⚡ Simulate Disruption at Selected Product",
            key=f"{section_key}_run_btn",
            use_container_width=True,
        )

        if run_btn:
            baseline = pruned_scores if is_pruned else dense_scores
            with st.spinner("Running disruption simulation…"):
                sim_result = simulate_disruption(
                    G_active, X, labels, prods, feat_cols,
                    prod_choice, baseline,
                )
            st.session_state[f"{section_key}_sim_result"] = sim_result
            st.rerun()

        if disrupt_state:
            sim = disrupt_state
            target_name = prods[sim["target_idx"]]

            st.markdown(f"**Simulated disruption at:** `{target_name}`")
            st.caption(f"Component size in current graph: {sim['component_size']} products")

            # ── Risk change table ─────────────────────────────────────────
            impacted = [sim["target_idx"]] + sim["hop1"] + sim["hop2"]
            rows = []
            for idx in impacted:
                role = "⚡ Disrupted" if idx == sim["target_idx"] \
                    else ("1-Hop" if idx in sim["hop1"] else "2-Hop")
                b = sim["before"][idx]
                a = sim["after"][idx]
                rows.append({
                    "Product":      prods[idx],
                    "Role":         role,
                    "Before Risk":  f"{b:.3f}",
                    "After Risk":   f"{a:.3f}",
                    "Δ Risk":       f"{a - b:+.3f}",
                })

            df_impact = pd.DataFrame(rows)
            st.dataframe(df_impact, use_container_width=True, hide_index=True)

            # ── Empty 2-hop message ───────────────────────────────────────
            if sim["empty_hop2"] and is_pruned:
                st.warning(sim["empty_hop2_msg"])
            elif sim["empty_hop2"]:
                st.info("No 2-hop neighbours found for this product in the current graph.")

            # Clear simulation
            if st.button("✕ Clear simulation", key=f"{section_key}_clear"):
                del st.session_state[f"{section_key}_sim_result"]
                st.rerun()

    # ── Model Inference Panel ────────────────────────────────────────────────
    st.subheader("📊 Model Inference & Validation Reference")
    m1, m2, m3, m4 = st.columns(4)
    graph_key = "pruned" if is_pruned else "dense"
    ref_metrics = PHASE1_METRICS[graph_key]

    m1.metric("ROC-AUC (Phase 1 LOOCV)",  f"{ref_metrics['ROC-AUC']:.4f}")
    m2.metric("PR-AUC (Phase 1 LOOCV)",   f"{ref_metrics['PR-AUC']:.4f}")
    m3.metric("Graph density",
              "17.4%" if is_pruned else "87.7%")
    m4.metric("Active edges",
              "136" if is_pruned else "684")

    st.caption(
        f"Predictions use **GraphSAGE** trained on the "
        f"{'pruned (w ≥ 3)' if is_pruned else 'dense (w ≥ 1)'} graph "
        f"with **18 non-leaking engineered features** (Production, Sales Orders, "
        f"Deliveries). Factory Issue features are excluded to prevent data leakage "
        f"from the label definition. Phase 1 metrics are from 40-fold LOOCV "
        f"on the 40-product SupplyGraph dataset."
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("⚠️ GraphRisk: Supply Chain Disruption Prediction Engine")
    st.markdown(
        "**Graph Neural Network–powered risk scoring across FMCG supply chains.** "
        "Products are modelled as graph nodes; shared production plants, product "
        "groups, and storage locations are modelled as weighted relational edges. "
        "A two-layer GraphSAGE model propagates operational signals across the "
        "network to identify products at risk of severe factory disruptions. "
        "Use the disruption simulator to explore how a production halt at one "
        "product cascades risk to connected neighbours."
    )

    st.warning(
        "**⚠️ Proxy Label Disclosure:** Risk scores are derived from an "
        "engineered proxy label — products were labelled 'at-risk' if their "
        "single largest day-over-day Factory Issue drop fell in the bottom 25th "
        "percentile across all 40 products. This is a quantitative heuristic "
        "designed for research benchmarking, **not** a verified enterprise "
        "record of supply chain failure. Interpret scores as disruption "
        "indicators, not authoritative ground truth."
    )

    # ── SECTION 1 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Section 1 — Fixed SupplyGraph Demo")
    st.markdown(
        "Explore the validated 40-product SupplyGraph dataset and simulate "
        "disruption scenarios using the pre-trained GraphSAGE model from Phase 1."
    )

    prods, df_feats, labels, q25, G_dense, G_pruned, X, feat_cols = load_sg_artifacts()

    with st.spinner("Computing dense graph risk scores…"):
        dense_scores  = cached_dense_scores(G_dense,  X, labels)
    with st.spinner("Computing pruned graph risk scores…"):
        pruned_scores = cached_pruned_scores(G_pruned, X, labels)

    render_graph_panel(
        prods, G_dense, G_pruned, X, labels, feat_cols,
        dense_scores, pruned_scores,
        section_key="s1",
    )

    # ── SECTION 2 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Section 2 — Bring Your Own Data")

    st.markdown("""
Upload your own supply chain data in the **same schema** as the SupplyGraph dataset.
The pipeline will validate your files, build a graph, compute proxy labels,
retrain a fresh GraphSAGE model, and render the full interactive dashboard for your data.

---
### Required file formats

| File | Required columns | Notes |
|------|-----------------|-------|
| **Nodes CSV** | `Node` | One row per product/SKU. No duplicates. |
| **Edge CSV(s)** | `node1`, `node2` (+ relation column) | One file per relation type (e.g. Plant, Storage). At least one edge file required. |
| **Factory Issue temporal CSV** | `Date`, then one column per product | Used to compute proxy labels. |
| **Other temporal CSVs** | `Date`, then one column per product | At least one of Production, Sales Order, or Delivery required for features. |

> **Not supported:** Arbitrary schemas, relational databases, JSON, XML,
> wide-format pivot tables without a Date column, or data with a different
> node/edge structure. This is a SupplyGraph-schema tool, not a
> general-purpose graph builder.
""")

    with st.expander("📁 Upload your data files", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            nodes_file = st.file_uploader(
                "Nodes CSV (required)", type="csv", key="s2_nodes")
            fi_file = st.file_uploader(
                "Factory Issue temporal CSV (required — used for labels)",
                type="csv", key="s2_fi")

        with col_b:
            edge_files_up = st.file_uploader(
                "Edge CSV(s) — one or more relation files",
                type="csv", accept_multiple_files=True, key="s2_edges")
            other_temp_files = st.file_uploader(
                "Other temporal CSVs (Production, Sales, Delivery…)",
                type="csv", accept_multiple_files=True, key="s2_temps")

    run_s2 = st.button("🚀 Build graph & train model on uploaded data",
                       key="s2_run", use_container_width=True)

    if run_s2:
        errors = []

        # ── Guard: required files present ──────────────────────────────────
        if nodes_file is None:
            errors.append("Nodes CSV is required.")
        if fi_file is None:
            errors.append("Factory Issue temporal CSV is required.")
        if not edge_files_up:
            errors.append("At least one edge CSV is required.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        try:
            # ── Parse and validate nodes ──────────────────────────────────
            df_nodes_up = pd.read_csv(nodes_file)
            prods_up    = validate_nodes_csv(df_nodes_up)
            N_up        = len(prods_up)

            st.info(f"Detected **{N_up} products** in your nodes file.")

            if N_up < 10:
                st.error(
                    f"Your dataset has only {N_up} products (< 10). "
                    "The model and graph visualization require at least 10 nodes."
                )
                st.stop()

            if N_up > 500:
                st.error(
                    f"Your dataset has {N_up} products (> 500). "
                    "LOOCV and Pyvis visualization do not scale to this size. "
                    "Please subsample to ≤ 500 products."
                )
                st.stop()

            # ── Parse and validate edges ──────────────────────────────────
            edges_dict_up = {}
            for ef in edge_files_up:
                df_e = pd.read_csv(ef)
                validate_edge_csv(df_e, prods_up, ef.name)
                edges_dict_up[ef.name] = df_e

            # ── Parse and validate Factory Issue temporal ─────────────────
            df_fi_up = pd.read_csv(fi_file)
            validate_temporal_csv(df_fi_up, prods_up, fi_file.name)

            # ── Parse and validate other temporal files ───────────────────
            temp_dfs_up = {("Factory_Issue", "Unit"): df_fi_up}

            feature_key_map = {
                "production": ("Production", "Unit"),
                "sales":      ("Sales_Order", "Unit"),
                "delivery":   ("Delivery",    "Unit"),
            }
            for tf in other_temp_files:
                df_t = pd.read_csv(tf)
                validate_temporal_csv(df_t, prods_up, tf.name)
                fname_lower = tf.name.lower()
                matched_key = None
                for kw, key in feature_key_map.items():
                    if kw in fname_lower:
                        matched_key = key
                        break
                if matched_key is None:
                    matched_key = (f"Custom_{tf.name[:20]}", "Unit")
                temp_dfs_up[matched_key] = df_t

            # ── Feature engineering ───────────────────────────────────────
            with st.spinner("Engineering node features…"):
                df_feats_up = engineer_features(temp_dfs_up, prods_up,
                                               drop_leaking=True,
                                               pop_dup_col="__none__")
                X_up       = df_feats_up.values.astype(np.float32)
                feat_cols_up = list(df_feats_up.columns)

            st.success(f"Features engineered: {df_feats_up.shape[0]} products × "
                       f"{df_feats_up.shape[1]} features")

            # ── Proxy labels ──────────────────────────────────────────────
            with st.spinner("Computing proxy labels from Factory Issue drops…"):
                labels_up, _, q25_up = compute_proxy_labels(
                    df_fi_up, prods_up, q=25.0, pop_dup_col="__none__"
                )

            is_skewed, maj_pct = check_label_skew(labels_up, threshold=0.85)
            n0 = int((labels_up == 0).sum())
            n1 = int((labels_up == 1).sum())
            st.info(f"Proxy label distribution: **{n0} Normal** / **{n1} At-Risk**  "
                    f"({100-maj_pct:.1f}% / {maj_pct:.1f}%)")

            if is_skewed:
                st.warning(
                    f"⚠️ Label distribution is heavily skewed "
                    f"({maj_pct:.1f}% majority class). "
                    "Evaluation metrics may be unreliable. "
                    "Consider using a different quantile threshold or "
                    "reviewing your Factory Issue data for constant products."
                )

            # ── Graph construction ────────────────────────────────────────
            with st.spinner("Building graph…"):
                G_dense_up  = build_graph(prods_up, edges_dict_up, df_feats_up)
                G_pruned_up = prune_graph(G_dense_up, min_weight=3)
                n_edges_d   = G_dense_up.number_of_edges()
                n_edges_p   = G_pruned_up.number_of_edges()

            st.success(f"Graph built: Dense = {n_edges_d} edges, "
                       f"Pruned (w≥3) = {n_edges_p} edges")

            # ── Retrain GraphSAGE ─────────────────────────────────────────
            st.warning(
                "**Model retrained on your uploaded data — results are specific "
                "to this dataset and are NOT directly comparable to the "
                "SupplyGraph benchmark numbers shown in Section 1.**"
            )

            protocol_used = ""
            with st.spinner("Training GraphSAGE model via cross-validation…"):
                in_ch_up = X_up.shape[1] if X_up.shape[1] > 0 else 1
                model_sg_up = SupplyGraphSAGEWrapper(in_ch_up)
                pyg_d_up    = graph_to_pyg(G_dense_up, labels_up)
                oof_up, metrics_up, protocol_used = evaluate_loocv(
                    pyg_d_up, X_up, labels_up)

            st.success(f"Training complete using {protocol_used}")
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC (your data)", f"{metrics_up['ROC-AUC']:.4f}")
            c2.metric("PR-AUC (your data)",  f"{metrics_up['PR-AUC']:.4f}")

            # ── Risk scores for display ───────────────────────────────────
            with st.spinner("Computing risk scores for visualisation…"):
                dense_scores_up  = get_full_risk_scores(G_dense_up,  X_up, labels_up)
                pruned_scores_up = get_full_risk_scores(G_pruned_up, X_up, labels_up)

            # ── Dashboard replay ──────────────────────────────────────────
            st.subheader("Your Data — Interactive Dashboard")
            render_graph_panel(
                prods_up, G_dense_up, G_pruned_up, X_up, labels_up, feat_cols_up,
                dense_scores_up, pruned_scores_up,
                section_key="s2",
            )

        except UploadError as e:
            st.error(f"❌ Validation error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            import traceback
            st.code(traceback.format_exc())


# ─── small shim so SupplyGraphSAGEWrapper alias works inside app.py ───────────
class SupplyGraphSAGEWrapper:
    """Thin alias — actual model is inside pipeline_core.evaluate_loocv."""
    def __init__(self, in_channels):
        self.in_channels = in_channels


if __name__ == "__main__":
    main()
