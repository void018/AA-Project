"""Streamlit dashboard for the open-phase detection experiments.

Reads ``runs_summary.json`` (one record per run) and ``runs_aggregate.json``
(one record per configuration, averaged over seeds), both produced by
``parse_runs.py``.

The dashboard leads with the aggregated view because a single run's
final-epoch test accuracy is not a stable quantity: for a configuration whose
validation curve oscillates, reruns of the identical setup can differ by more
than the gap being claimed between two feature sets. Mean +- std over seeds,
alongside an explicit stability column, is the honest summary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OPC Detection — Experiment Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUMMARY_FILE = "runs_summary.json"
AGGREGATE_FILE = "runs_aggregate.json"

# A configuration whose validation accuracy swings by more than this within the
# last 50 epochs is flagged: its final-epoch score is largely luck.
SPREAD_WARN = 0.05
SPREAD_BAD = 0.15


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_json(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load a JSON list from *path*, or return None if unavailable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to parse '{path}': {exc}")
        return None


def stability_label(spread: Optional[float]) -> str:
    """Turn a last-50-epoch spread into a readable stability verdict."""
    if spread is None:
        return "unknown"
    if spread >= SPREAD_BAD:
        return "unstable"
    if spread >= SPREAD_WARN:
        return "marginal"
    return "stable"


# ── Display helpers ──────────────────────────────────────────────────────────

def display_confusion_matrix(cm_data: List[List[int]]) -> None:
    """Render a 2x2 confusion matrix [[TN, FP], [FN, TP]] as a table."""
    st.table(pd.DataFrame(
        data=cm_data,
        columns=["Predicted Healthy", "Predicted Faulted"],
        index=["True Healthy", "True Faulted"],
    ))


def display_classification_report(report_data: Dict[str, Any]) -> None:
    """Render the structured classification report, excluding the scalar accuracy."""
    rows = {k: v for k, v in report_data.items() if isinstance(v, dict)}
    if not rows:
        st.warning("No per-class metrics found.")
        return
    df = pd.DataFrame(rows).transpose()
    for col in ("precision", "recall", "f1-score"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")
    if "support" in df.columns:
        df["support"] = df["support"].astype(int)
    st.table(df)


def aggregate_frame(aggregates: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build the per-configuration table shown on the overview tab."""
    df = pd.DataFrame(aggregates)
    if df.empty:
        return df
    df["stability"] = df["val_spread_worst"].apply(stability_label)
    df["test_acc"] = df.apply(
        lambda r: f"{r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}"
        if r["test_acc_mean"] is not None else "—",
        axis=1,
    )
    return df


# ── Tabs ─────────────────────────────────────────────────────────────────────

def tab_overview(agg_df: pd.DataFrame) -> None:
    """Per-configuration comparison, averaged over seeds."""
    st.subheader("Configurations, averaged across seeds")

    if agg_df.empty:
        st.warning("No configurations to display.")
        return

    unstable = agg_df[agg_df["stability"] == "unstable"]
    if not unstable.empty:
        st.warning(
            f"{len(unstable)} configuration(s) show a validation-accuracy swing "
            f"of ≥{SPREAD_BAD:.0%} across the last 50 epochs. For these, the "
            "final-epoch score is close to a coin flip — quote the mean ± std "
            "and the best checkpoint instead, and treat rank ordering with care."
        )

    cols = [
        "config_name", "n_runs", "num_features", "test_acc",
        "test_acc_min", "test_acc_max", "best_val_acc_mean",
        "healthy_recall_mean", "faulted_recall_mean",
        "val_spread_worst", "stability",
    ]
    st.dataframe(
        agg_df[[c for c in cols if c in agg_df.columns]],
        column_config={
            "config_name": st.column_config.TextColumn("Configuration"),
            "n_runs": st.column_config.NumberColumn("Seeds", width="small"),
            "num_features": st.column_config.NumberColumn("# Feat", width="small"),
            "test_acc": st.column_config.TextColumn("Test acc (mean ± std)"),
            "test_acc_min": st.column_config.NumberColumn("Min", format="%.4f"),
            "test_acc_max": st.column_config.NumberColumn("Max", format="%.4f"),
            "best_val_acc_mean": st.column_config.NumberColumn("Best val", format="%.4f"),
            "healthy_recall_mean": st.column_config.NumberColumn("Healthy rec.", format="%.2f"),
            "faulted_recall_mean": st.column_config.NumberColumn("Faulted rec.", format="%.2f"),
            "val_spread_worst": st.column_config.NumberColumn("Worst spread", format="%.4f"),
            "stability": st.column_config.TextColumn("Stability"),
        },
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Mean test accuracy by configuration")
    chart_df = agg_df.set_index("config_name")[["test_acc_mean"]].dropna()
    if not chart_df.empty:
        st.bar_chart(chart_df, height=340)

    st.markdown("#### Seed-to-seed spread")
    st.caption(
        "Distance between the best and worst seed for each configuration. "
        "A tall bar means the single-run number is not reproducible."
    )
    spread_df = agg_df.dropna(subset=["test_acc_min", "test_acc_max"]).copy()
    if not spread_df.empty:
        spread_df["seed_range"] = spread_df["test_acc_max"] - spread_df["test_acc_min"]
        st.bar_chart(
            spread_df.set_index("config_name")[["seed_range"]], height=300
        )


def tab_runs(runs_df: pd.DataFrame) -> None:
    """Every individual run, including seed and stability columns."""
    st.subheader("Individual runs")
    st.caption(
        "One row per run. 'Final vs best' is how much accuracy was lost by "
        "reporting the last epoch rather than the best checkpoint."
    )

    display = runs_df.copy()
    display["test_accuracy"] = display["test_results"].apply(
        lambda x: x.get("accuracy") if isinstance(x, dict) else None
    )
    display["stability"] = display["val_acc_last50_spread"].apply(stability_label)

    cols = [
        "run_codename", "config_name", "seed", "num_features",
        "test_accuracy", "best_val_acc", "last_val_acc",
        "val_acc_last50_spread", "final_vs_best_gap",
        "best_epoch", "epochs_completed", "early_stopped_at", "stability",
    ]
    st.dataframe(
        display[[c for c in cols if c in display.columns]],
        column_config={
            "run_codename": st.column_config.TextColumn("Run"),
            "config_name": st.column_config.TextColumn("Configuration"),
            "seed": st.column_config.NumberColumn("Seed", width="small"),
            "num_features": st.column_config.NumberColumn("# Feat", width="small"),
            "test_accuracy": st.column_config.NumberColumn("Test acc", format="%.4f"),
            "best_val_acc": st.column_config.NumberColumn("Best val", format="%.4f"),
            "last_val_acc": st.column_config.NumberColumn("Final val", format="%.4f"),
            "val_acc_last50_spread": st.column_config.NumberColumn("Last-50 spread", format="%.4f"),
            "final_vs_best_gap": st.column_config.NumberColumn("Final vs best", format="%.4f"),
            "best_epoch": st.column_config.NumberColumn("Best ep.", width="small"),
            "epochs_completed": st.column_config.NumberColumn("Epochs", width="small"),
            "early_stopped_at": st.column_config.NumberColumn("Stopped", width="small"),
        },
        width="stretch",
        hide_index=True,
    )


def tab_detail(runs_df: pd.DataFrame) -> None:
    """Drill into one run: metrics, settings, curves, confusion matrix, report."""
    st.subheader("Inspect a single run")

    configs = sorted(runs_df["config_name"].dropna().unique())
    if not configs:
        st.warning("No runs available.")
        return

    left, right = st.columns([2, 1])
    with left:
        config_name = st.selectbox("Configuration", options=configs)
    subset = runs_df[runs_df["config_name"] == config_name]
    with right:
        seeds = sorted(s for s in subset["seed"].dropna().unique())
        seed = st.selectbox("Seed", options=seeds) if seeds else None

    row = subset[subset["seed"] == seed] if seed is not None else subset
    if row.empty:
        st.warning("No run matches that selection.")
        return
    run = row.iloc[0]

    test_acc = (run["test_results"] or {}).get("accuracy")
    spread = run.get("val_acc_last50_spread")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{test_acc:.4f}" if test_acc is not None else "—")
    m2.metric(
        "Best val accuracy",
        f"{run['best_val_acc']:.4f}" if run.get("best_val_acc") is not None else "—",
        delta=(f"{-run['final_vs_best_gap']:.4f} at final epoch"
               if run.get("final_vs_best_gap") else None),
        delta_color="inverse",
    )
    m3.metric("Best epoch", run.get("best_epoch") or "—")
    m4.metric(
        "Last-50 spread",
        f"{spread:.4f}" if spread is not None else "—",
        help="Validation-accuracy range over the final 50 epochs. "
             "Large values mean the final-epoch score is not reproducible.",
    )

    if spread is not None and spread >= SPREAD_BAD:
        st.error(
            f"This run's validation accuracy swung by {spread:.2%} over its last "
            "50 epochs, so its final-epoch score should not be quoted on its own."
        )
    elif spread is not None and spread >= SPREAD_WARN:
        st.warning(
            f"Moderate instability ({spread:.2%} swing over the last 50 epochs)."
        )

    st.markdown("**Pipeline settings for this run**")
    settings = {
        "Log-transform": run.get("log_transform"),
        "Dropout": run.get("dropout"),
        "LR monitor": run.get("lr_monitor"),
        "Best checkpoint used": run.get("use_best_checkpoint"),
        "Early stopped at": run.get("early_stopped_at"),
    }
    known = {k: v for k, v in settings.items() if v is not None}
    if known:
        st.table(pd.DataFrame([known]))
    else:
        st.info(
            "This run predates the settings logging — re-run it for comparable numbers."
        )

    st.markdown(f"**Features ({run.get('num_features', 0)}):**")
    st.info(f"`{run['features_used']}`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training curves**")
        path = run.get("training_curves_path")
        if path and os.path.exists(path):
            st.image(path)
        else:
            st.warning("Training curves image not found.")
    with c2:
        st.markdown("**Confusion matrix**")
        if run.get("confusion_matrix"):
            display_confusion_matrix(run["confusion_matrix"])
        else:
            st.warning("No confusion matrix data found.")

    st.markdown("**Classification report**")
    if run.get("test_results"):
        display_classification_report(run["test_results"])
    else:
        st.warning("No classification report data found.")


def tab_seed_compare(runs_df: pd.DataFrame) -> None:
    """Show how far apart seeds of the same configuration land."""
    st.subheader("Seed comparison")
    st.caption(
        "Test accuracy of each seed, per configuration. Points that scatter "
        "widely within one configuration mean any single run of it is unreliable."
    )

    df = runs_df.copy()
    df["test_accuracy"] = df["test_results"].apply(
        lambda x: x.get("accuracy") if isinstance(x, dict) else None
    )
    df = df.dropna(subset=["test_accuracy", "config_name", "seed"])
    if df.empty:
        st.info("No multi-seed runs found yet. Set config.SEEDS and re-run.")
        return

    pivot = df.pivot_table(
        index="config_name", columns="seed",
        values="test_accuracy", aggfunc="first",
    )
    pivot.columns = [f"seed {int(c)}" for c in pivot.columns]
    pivot["range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot = pivot.sort_values("range", ascending=False)

    st.dataframe(
        pivot.style.format("{:.4f}"),
        width="stretch",
    )
    st.bar_chart(pivot[["range"]], height=320)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("⚡ Open-Phase Detection — Experiment Dashboard")

    runs = load_json(SUMMARY_FILE)
    aggregates = load_json(AGGREGATE_FILE)

    if runs is None:
        st.error(
            f"'{SUMMARY_FILE}' not found. Run `python parse_runs.py` first."
        )
        return
    if not runs:
        st.warning("No runs parsed yet.")
        return

    runs_df = pd.DataFrame(runs)
    for col in ("config_name", "seed", "num_features", "best_val_acc",
                "val_acc_last50_spread", "final_vs_best_gap", "best_epoch",
                "epochs_completed", "early_stopped_at", "log_transform",
                "dropout", "lr_monitor", "use_best_checkpoint"):
        if col not in runs_df.columns:
            runs_df[col] = None

    if aggregates is None:
        st.info(
            f"'{AGGREGATE_FILE}' not found — re-run `parse_runs.py` to generate it. "
            "Falling back to per-run aggregation."
        )
        from parse_runs import aggregate_by_config
        aggregates = aggregate_by_config(runs)

    # ── Sidebar filters ──────────────────────────────────────────────────────
    st.sidebar.header("Filters")

    min_acc = st.sidebar.slider(
        "Minimum mean test accuracy", 0.0, 1.0, 0.0, 0.01
    )
    hide_unstable = st.sidebar.checkbox(
        "Hide unstable configurations",
        value=False,
        help=f"Hides configurations whose validation accuracy swings by "
             f"≥{SPREAD_BAD:.0%} over the last 50 epochs.",
    )
    all_features = sorted({f for r in runs for f in (r.get("features_used") or [])})
    selected_features = st.sidebar.multiselect(
        "Must contain features", options=all_features,
        help="Show only configurations that include ALL selected features.",
    )

    agg_df = aggregate_frame(aggregates)
    if not agg_df.empty:
        agg_df = agg_df[agg_df["test_acc_mean"].fillna(0) >= min_acc]
        if hide_unstable:
            agg_df = agg_df[agg_df["stability"] != "unstable"]
        if selected_features:
            agg_df = agg_df[agg_df["features_used"].apply(
                lambda fs: all(f in fs for f in selected_features)
            )]

    keep = set(agg_df["config_name"]) if not agg_df.empty else set()
    filtered_runs = runs_df[runs_df["config_name"].isin(keep)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Configurations", len(agg_df))
    st.sidebar.metric("Total runs", len(filtered_runs))
    stale = runs_df["best_val_acc"].isna().sum()
    if stale:
        st.sidebar.warning(f"{stale} run(s) predate stability logging.")

    if agg_df.empty:
        st.warning("No configurations match the current filters.")
        return

    overview, per_run, seeds_tab, detail = st.tabs(
        ["Overview", "All runs", "Seed comparison", "Run detail"]
    )
    with overview:
        tab_overview(agg_df)
    with per_run:
        tab_runs(filtered_runs)
    with seeds_tab:
        tab_seed_compare(filtered_runs)
    with detail:
        tab_detail(filtered_runs)


if __name__ == "__main__":
    main()