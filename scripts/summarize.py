import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import json
    from pathlib import Path
    from typing import Any, Dict, List

    import numpy as np
    import pandas as pd

    return Any, Dict, List, Path, argparse, json, np, pd


@app.cell
def _(Any, Dict, List, Path, json, pd):
    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


    def safe_mean(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.mean())



    return read_jsonl, safe_mean


@app.cell
def _(Path, argparse, json, np, pd, read_jsonl, safe_mean):
    def main() -> None:
        parser = argparse.ArgumentParser(description="Summarize Attempt 1 inference outputs.")
        parser.add_argument("--in-file", default="att1/output/responses.jsonl")
        parser.add_argument("--out-csv", default="att1/output/summary.csv")
        parser.add_argument("--out-json", default="att1/output/summary.json")
        args = parser.parse_args()

        rows = read_jsonl(Path(args.in_file))
        if not rows:
            raise RuntimeError("No rows found in input file.")

        df = pd.DataFrame(rows)
        df["exact_match"] = df["exact_match"].astype(int)
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

        summary_rows = []
        for (dataset, template_name), group in df.groupby(["dataset", "template_name"]):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "template_name": template_name,
                    "n": int(len(group)),
                    "exact_match": safe_mean(group["exact_match"]),
                    "f1": safe_mean(group["f1"].fillna(0.0)),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "template_name"])

        dataset_summary = (
            df.groupby("dataset", as_index=False)
            .agg(
                n=("exact_match", "size"),
                exact_match=("exact_match", "mean"),
                f1=("f1", "mean"),
            )
            .fillna(0.0)
        )

        item_summary = (
            df.groupby(["dataset", "id"], as_index=False)
            .agg(
                exact_match_std=("exact_match", lambda x: float(np.std(x, ddof=0))),
                exact_match_mean=("exact_match", "mean"),
            )
            .fillna(0.0)
        )

        sensitivity_summary = (
            item_summary.groupby("dataset", as_index=False)["exact_match_std"]
            .mean()
            .rename(columns={"exact_match_std": "mean_item_std"})
        )

        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_csv = Path("att1/data/raw_results/inference_results.csv")
        raw_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(raw_csv, index=False)
        summary_df.to_csv(out_csv, index=False)
        dataset_summary.to_csv(out_csv.with_name("att1_dataset_summary.csv"), index=False)
        item_summary.to_csv(out_csv.with_name("att1_item_sensitivity.csv"), index=False)
        sensitivity_summary.to_csv(out_csv.with_name("att1_dataset_sensitivity.csv"), index=False)

        payload = {
            "n_rows": int(len(df)),
            "dataset_summary": dataset_summary.to_dict(orient="records"),
            "dataset_sensitivity": sensitivity_summary.to_dict(orient="records"),
        }

        with Path(args.out_json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        print(f"Saved raw inference rows to {raw_csv}")
        print(f"Saved template summary to {out_csv}")
        print(f"Saved dataset summary to {out_csv.with_name('att1_dataset_summary.csv')}")
        print(f"Saved item sensitivity to {out_csv.with_name('att1_item_sensitivity.csv')}")
        print(f"Saved dataset sensitivity to {out_csv.with_name('att1_dataset_sensitivity.csv')}")


    if __name__ == "__main__":
        main()
    return


if __name__ == "__main__":
    app.run()
