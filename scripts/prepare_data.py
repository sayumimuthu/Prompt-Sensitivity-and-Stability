import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import json
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Dict, List, Sequence

    from datasets import load_dataset

    return (
        Any,
        Dict,
        List,
        Path,
        Sequence,
        argparse,
        dataclass,
        json,
        load_dataset,
    )


@app.cell
def _(Any, Dict, List, Path, Sequence, dataclass, json):
    @dataclass
    class SampleConfig:
        dataset_name: str
        split: str
        task_type: str
        num_items: int


    def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


    def sample_indices(length: int, count: int, seed: int) -> List[int]:
        import random

        rng = random.Random(seed)
        indices = list(range(length))
        rng.shuffle(indices)
        return indices[: min(count, length)]


    def normalize_arc(example: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
        choices = list(example["choices"]["text"])
        labels = list(example["choices"]["label"])
        answer_key = example.get("answerKey", "")
        answer_index = labels.index(answer_key) if answer_key in labels else 0
        options = [f"{label}) {text}" for label, text in zip(labels, choices)]
        return {
            "id": f"{dataset_name}_{idx:04d}",
            "dataset": dataset_name,
            "task_type": "mcq",
            "question": example["question"],
            "options": options,
            "option_labels": labels,
            "gold_answer": labels[answer_index],
            "gold_text": choices[answer_index],
            "source": "huggingface",
        }


    def normalize_boolq(example: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
        answer = "yes" if bool(example["answer"]) else "no"
        return {
            "id": f"{dataset_name}_{idx:04d}",
            "dataset": dataset_name,
            "task_type": "boolean",
            "question": example["question"],
            "context": example.get("passage", ""),
            "gold_answer": answer,
            "source": "huggingface",
        }


    def normalize_squad(example: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
        answers = example.get("answers", {}).get("text", [])
        gold_answer = answers[0] if answers else ""
        return {
            "id": f"{dataset_name}_{idx:04d}",
            "dataset": dataset_name,
            "task_type": "qa",
            "question": example["question"],
            "context": example["context"],
            "gold_answer": gold_answer,
            "source": "huggingface",
        }



    return (
        SampleConfig,
        normalize_arc,
        normalize_boolq,
        normalize_squad,
        sample_indices,
        write_jsonl,
    )


@app.cell
def _(
    Any,
    Dict,
    List,
    SampleConfig,
    load_dataset,
    normalize_arc,
    normalize_boolq,
    normalize_squad,
    sample_indices,
):
    def load_and_normalize(config: SampleConfig, seed: int) -> List[Dict[str, Any]]:
        if config.dataset_name in {"arc_easy", "arc_challenge"}:
            dataset = load_dataset("ai2_arc", "ARC-Easy" if config.dataset_name == "arc_easy" else "ARC-Challenge", split=config.split)
            selector = sample_indices(len(dataset), config.num_items, seed)
            return [normalize_arc(dataset[i], config.dataset_name, n + 1) for n, i in enumerate(selector)]

        if config.dataset_name == "boolq":
            dataset = load_dataset("boolq", split=config.split)
            selector = sample_indices(len(dataset), config.num_items, seed)
            return [normalize_boolq(dataset[i], config.dataset_name, n + 1) for n, i in enumerate(selector)]

        if config.dataset_name == "squad":
            dataset = load_dataset("squad", split=config.split)
            selector = sample_indices(len(dataset), config.num_items, seed)
            return [normalize_squad(dataset[i], config.dataset_name, n + 1) for n, i in enumerate(selector)]

        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    return (load_and_normalize,)


@app.cell
def _(Path, SampleConfig, argparse, json, load_and_normalize, write_jsonl):
    def main() -> None:
        parser = argparse.ArgumentParser(description="Download and normalize benchmark datasets for Attempt 1.")
        parser.add_argument("--num-items", type=int, default=15)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--out-dir", default="att1/data")
        parser.add_argument("--arc-split", default="test")
        parser.add_argument("--boolq-split", default="validation")
        parser.add_argument("--squad-split", default="validation")
        args = parser.parse_args()

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        datasets_to_build = [
            SampleConfig("arc_easy", args.arc_split, "mcq", args.num_items),
            SampleConfig("arc_challenge", args.arc_split, "mcq", args.num_items),
            SampleConfig("boolq", args.boolq_split, "boolean", args.num_items),
            SampleConfig("squad", args.squad_split, "qa", args.num_items),
        ]

        summary = []
        for config in datasets_to_build:
            rows = load_and_normalize(config, args.seed)
            write_jsonl(out_dir / f"{config.dataset_name}.jsonl", rows)
            summary.append({"dataset": config.dataset_name, "n": len(rows), "split": config.split, "task_type": config.task_type})
            print(f"Wrote {len(rows)} records to {out_dir / f'{config.dataset_name}.jsonl'}")

        with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


    if __name__ == "__main__":
        main()
    return


if __name__ == "__main__":
    app.run()
