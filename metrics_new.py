import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    '''
    Compute semantic stability and artifact scores from Attempt 1 outputs.

    Supports both:
    - att1/data/raw_results/inference_results.csv
    - att1/output/responses.jsonl
    '''

    import argparse
    import json
    from pathlib import Path
    from typing import Callable, Dict, List, Sequence, Tuple

    import numpy as np
    import pandas as pd


    return Callable, Dict, List, Path, Sequence, Tuple, argparse, json, np, pd


@app.cell
def _(Dict, List, Path, json, np, pd):
    def read_jsonl(path: Path) -> pd.DataFrame:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)


    def load_input(input_path: Path) -> pd.DataFrame:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() == ".jsonl":
            df = read_jsonl(input_path)
        else:
            df = pd.read_csv(input_path)

        return df


    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "item_id" not in df.columns and "id" in df.columns:
            df["item_id"] = df["id"]
        if "response" not in df.columns and "raw_response" in df.columns:
            df["response"] = df["raw_response"]
        if "is_correct" not in df.columns and "exact_match" in df.columns:
            df["is_correct"] = df["exact_match"]
        if "backend" not in df.columns:
            df["backend"] = "unknown"

        required = ["dataset", "item_id", "template_name", "model", "response", "is_correct"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["response"] = df["response"].astype(str)
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)

        if "f1" in df.columns:
            df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        else:
            df["f1"] = np.nan

        return df


    return load_input, normalize_columns


@app.cell
def _(Callable, Sequence, Tuple, np):
    def get_similarity_function(prefer_embeddings: bool = True) -> Tuple[Callable[[Sequence[str]], np.ndarray], str]:
        def token_jaccard_matrix(texts: Sequence[str]) -> np.ndarray:
            n = len(texts)
            out = np.zeros((n, n), dtype=float)
            token_sets = [set(str(t).lower().split()) for t in texts]
            for i in range(n):
                out[i, i] = 1.0
                for j in range(i + 1, n):
                    union = token_sets[i] | token_sets[j]
                    if not union:
                        sim = 1.0
                    else:
                        sim = len(token_sets[i] & token_sets[j]) / len(union)
                    out[i, j] = sim
                    out[j, i] = sim
            return out

        if not prefer_embeddings:
            return token_jaccard_matrix, "token_jaccard"

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            def embedding_cosine_matrix(texts: Sequence[str]) -> np.ndarray:
                embeddings = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-8, norms)
                normalized = embeddings / norms
                sims = normalized @ normalized.T
                sims = np.clip(sims, -1.0, 1.0)
                sims = (sims + 1.0) / 2.0
                return np.clip(sims, 0.0, 1.0)

            return embedding_cosine_matrix, "sentence_transformer_all-MiniLM-L6-v2"
        except Exception:
            return token_jaccard_matrix, "token_jaccard"



    return (get_similarity_function,)


@app.cell
def _(Callable, Dict, List, Sequence, np, pd):
    def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] < 2:
            return np.array([], dtype=float)
        idx = np.triu_indices(matrix.shape[0], k=1)
        return matrix[idx]


    def compute_item_level_metrics(
        df: pd.DataFrame,
        similarity_matrix_fn: Callable[[Sequence[str]], np.ndarray],
        sens_threshold: float,
        stab_high_threshold: float,
        stab_low_threshold: float,
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        group_keys = ["dataset", "item_id", "model", "backend"]

        for keys, group in df.groupby(group_keys, dropna=False):
            dataset, item_id, model, backend = keys
            templates = int(group["template_name"].nunique())

            is_correct = group["is_correct"].astype(float).to_numpy()
            sens_acc = float(np.std(is_correct, ddof=0))
            mean_acc = float(np.mean(is_correct))

            f1_vals = group["f1"].dropna().astype(float).to_numpy()
            f1_sens = float(np.std(f1_vals, ddof=0)) if f1_vals.size > 0 else np.nan
            mean_f1 = float(np.mean(f1_vals)) if f1_vals.size > 0 else np.nan

            responses = group["response"].fillna("").astype(str).tolist()
            sim_matrix = similarity_matrix_fn(responses)
            pair_vals = upper_triangle_values(sim_matrix)

            stab_sem = float(np.mean(pair_vals)) if pair_vals.size > 0 else 1.0
            stab_sem_std = float(np.std(pair_vals, ddof=0)) if pair_vals.size > 0 else 0.0

            # New decomposition:
            # EvalArtifactScore high when metric sensitivity is high but semantics stay stable.
            eval_artifact_score = float(sens_acc * stab_sem)

            # RealInstabilityScore high when metric sensitivity is high and semantics drift.
            real_instability_score = float(sens_acc * (1.0 - stab_sem))

            if sens_acc < sens_threshold:
                category = "STABLE_OR_LOW_VARIANCE"
            elif stab_sem >= stab_high_threshold:
                category = "LIKELY_EVAL_ARTIFACT"
            elif stab_sem <= stab_low_threshold:
                category = "LIKELY_REAL_INSTABILITY"
            else:
                category = "MIXED"

            rows.append(
                {
                    "dataset": dataset,
                    "item_id": item_id,
                    "model": model,
                    "backend": backend,
                    "num_templates": templates,
                    "SensAcc": sens_acc,
                    "StabSem": stab_sem,
                    "StabSemStd": stab_sem_std,
                    "mean_accuracy": mean_acc,
                    "mean_f1": mean_f1,
                    "F1Sens": f1_sens,
                    "EvalArtifactScore": eval_artifact_score,
                    "RealInstabilityScore": real_instability_score,
                    "Category": category,
                }
            )

        return pd.DataFrame(rows)

    return (compute_item_level_metrics,)


@app.cell
def _(List, Path, pd):
    def summarize_by(item_df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
        return (
            item_df.groupby(by_cols, as_index=False)
            .agg(
                n_items=("item_id", "nunique"),
                mean_sens_acc=("SensAcc", "mean"),
                mean_stab_sem=("StabSem", "mean"),
                mean_eval_artifact=("EvalArtifactScore", "mean"),
                mean_real_instability=("RealInstabilityScore", "mean"),
                mean_accuracy=("mean_accuracy", "mean"),
                mean_f1=("mean_f1", "mean"),
            )
            .fillna(0.0)
        )


    def save_outputs(item_df: pd.DataFrame, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        item_file = out_dir / "metrics_summary.csv"
        model_file = out_dir / "model_metrics_summary.csv"
        dataset_file = out_dir / "dataset_metrics_summary.csv"
        category_file = out_dir / "category_breakdown.csv"

        item_df.to_csv(item_file, index=False)
        summarize_by(item_df, ["model", "backend"]).to_csv(model_file, index=False)
        summarize_by(item_df, ["dataset"]).to_csv(dataset_file, index=False)

        category = (
            item_df.groupby(["model", "backend", "Category"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        category.to_csv(category_file, index=False)

        print(f"Saved item-level metrics: {item_file}")
        print(f"Saved model summary: {model_file}")
        print(f"Saved dataset summary: {dataset_file}")
        print(f"Saved category breakdown: {category_file}")



    return (save_outputs,)


@app.cell
def _(Path, pd):
    def create_plots(item_df: pd.DataFrame, figures_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping plots.")
            return

        figures_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Scatter of individual items colored by dataset & category.
        fig, ax = plt.subplots(figsize=(10, 7))
    
        # Use category for coloring to show interpretation zones visually
        category_colors = {
            "STABLE_OR_LOW_VARIANCE": "#2ecc71",      # Green
            "LIKELY_EVAL_ARTIFACT": "#f39c12",         # Orange
            "LIKELY_REAL_INSTABILITY": "#e74c3c",     # Red
        }
    
        for category in item_df["Category"].unique():
            mask = item_df["Category"] == category
            subset = item_df[mask]
            ax.scatter(
                subset["SensAcc"],
                subset["StabSem"],
                alpha=0.6,
                s=60,
                label=category,
                color=category_colors.get(category, "#3498db"),
            )
    
        # Threshold lines for interpretation zones
        ax.axhline(0.75, linestyle="--", linewidth=1.5, color="gray", alpha=0.5, label="StabSem thresholds")
        ax.axhline(0.45, linestyle=":", linewidth=1.5, color="gray", alpha=0.5)
        ax.axvline(0.15, linestyle="--", linewidth=1.5, color="gray", alpha=0.5, label="SensAcc threshold")
    
        ax.set_xlabel("Accuracy Sensitivity (SensAcc)", fontsize=11)
        ax.set_ylabel("Semantic Stability (StabSem)", fontsize=11)
        ax.set_title("Item-Level Sensitivity vs Semantic Stability (n={:d})".format(len(item_df)), fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(-0.05, max(0.6, item_df["SensAcc"].max() + 0.05))
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "sensitivity_vs_stability.png", dpi=180)
        plt.close(fig)

        # Plot 2: Category distribution as bar chart (more meaningful than tiny artifact scores).
        category_counts = item_df["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [category_colors.get(cat, "#3498db") for cat in category_counts.index]
        bars = ax.bar(range(len(category_counts)), category_counts.values, color=colors, alpha=0.75, edgecolor="black", linewidth=1.5)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=15, ha="right")
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Distribution of Items by Stability Category", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
    
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=10)
    
        fig.tight_layout()
        fig.savefig(figures_dir / "category_distribution.png", dpi=180)
        plt.close(fig)

        # Plot 3: Box plots of SensAcc and StabSem by dataset (shows variation).
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
        datasets = sorted(item_df["dataset"].unique())
        sens_data = [item_df[item_df["dataset"] == ds]["SensAcc"].values for ds in datasets]
        stab_data = [item_df[item_df["dataset"] == ds]["StabSem"].values for ds in datasets]
    
        ax1.boxplot(sens_data, labels=datasets)
        ax1.set_ylabel("Accuracy Sensitivity", fontsize=10)
        ax1.set_title("SensAcc Distribution by Dataset", fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")
    
        ax2.boxplot(stab_data, labels=datasets)
        ax2.set_ylabel("Semantic Stability", fontsize=10)
        ax2.set_title("StabSem Distribution by Dataset", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")
    
        fig.tight_layout()
        fig.savefig(figures_dir / "metrics_by_dataset.png", dpi=180)
        plt.close(fig)

        print(f"Saved plots to {figures_dir}")



    return (create_plots,)


@app.cell
def _(
    Path,
    argparse,
    compute_item_level_metrics,
    create_plots,
    get_similarity_function,
    load_input,
    normalize_columns,
    save_outputs,
):
    def main() -> None:
        parser = argparse.ArgumentParser(description="Compute Attempt 1 stability and artifact metrics.")
        parser.add_argument(
            "--input-file",
            default="att1/data/raw_results/inference_results.csv",
            help="Path to Attempt 1 output file. Supports CSV and JSONL.",
        )
        parser.add_argument("--out-dir", default="att1/data/raw_results")
        parser.add_argument("--figures-dir", default="att1/analysis/figures")
        parser.add_argument("--no-embeddings", action="store_true", help="Use token Jaccard instead of embeddings.")
        parser.add_argument("--sens-threshold", type=float, default=0.15)
        parser.add_argument("--stab-high-threshold", type=float, default=0.75)
        parser.add_argument("--stab-low-threshold", type=float, default=0.45)
        args = parser.parse_args()

        input_path = Path(args.input_file)
        if not input_path.exists() and input_path.name == "inference_results.csv":
            fallback = Path("att1/output/responses.jsonl")
            if fallback.exists():
                input_path = fallback

        raw_df = load_input(input_path)
        df = normalize_columns(raw_df)

        print("Loaded Attempt 1 results")
        print(f"  Source: {input_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Datasets: {sorted(df['dataset'].unique().tolist())}")
        print(f"  Models: {sorted(df['model'].unique().tolist())}")
        print(f"  Templates: {df['template_name'].nunique()}")

        sim_fn, sim_name = get_similarity_function(prefer_embeddings=not args.no_embeddings)
        print(f"Semantic similarity backend: {sim_name}")

        item_metrics = compute_item_level_metrics(
            df,
            similarity_matrix_fn=sim_fn,
            sens_threshold=args.sens_threshold,
            stab_high_threshold=args.stab_high_threshold,
            stab_low_threshold=args.stab_low_threshold,
        )

        print("\nCategory distribution:")
        print(item_metrics["Category"].value_counts())

        corr = float(item_metrics["SensAcc"].corr(item_metrics["StabSem"]))
        print(f"\nCorrelation(SensAcc, StabSem): {corr:.4f}")

        save_outputs(item_metrics, Path(args.out_dir))
        create_plots(item_metrics, Path(args.figures_dir))

        print("\nAttempt 1 metrics complete.")


    if __name__ == "__main__":
        main()

    return


if __name__ == "__main__":
    app.run()
