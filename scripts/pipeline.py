import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    '''
    Usage:
        python scripts/pipeline.py --backend ollama --model llama3.1:8b
        python scripts/pipeline.py --backend hf_api --model HuggingFaceTB/SmolLM2-1.7B-Instruct
        python scripts/pipeline.py --backend openai --model gpt-4o-mini

    Steps:
        1. Download & normalize benchmark datasets (ARC, BoolQ, SQuAD)
        2. Generate 8 prompt template variants per item
        3. Run inference via chosen backend
        4. Summarize results into CSVs
        5. Compute SensAcc / StabSem / artifact metrics and plots
    '''
    return


@app.cell
def _():
    import argparse
    import json
    import os
    import random
    import re
    from pathlib import Path
    from typing import Any, Dict, List, Sequence

    #1. Prepare data

    def sample_indices(length: int, count: int, seed: int) -> List[int]:
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
            "gold_answer": labels[answer_index],
            "gold_text": choices[answer_index],
        }


    def normalize_boolq(example: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
        return {
            "id": f"{dataset_name}_{idx:04d}",
            "dataset": dataset_name,
            "task_type": "boolean",
            "question": example["question"],
            "context": example.get("passage", ""),
            "gold_answer": "yes" if bool(example["answer"]) else "no",
        }


    def normalize_squad(example: Dict[str, Any], dataset_name: str, idx: int) -> Dict[str, Any]:
        answers = example.get("answers", {}).get("text", [])
        return {
            "id": f"{dataset_name}_{idx:04d}",
            "dataset": dataset_name,
            "task_type": "qa",
            "question": example["question"],
            "context": example["context"],
            "gold_answer": answers[0] if answers else "",
        }


    def prepare_data(out_dir: Path, num_items: int, seed: int) -> None:
        from datasets import load_dataset

        configs = [
            ("arc_easy", "ai2_arc", "ARC-Easy", "test", normalize_arc),
            ("arc_challenge", "ai2_arc", "ARC-Challenge", "test", normalize_arc),
            ("boolq", "boolq", None, "validation", normalize_boolq),
            ("squad", "squad", None, "validation", normalize_squad),
        ]

        out_dir.mkdir(parents=True, exist_ok=True)
        summary = []

        for dataset_name, hf_name, config_name, split, normalizer in configs:
            if config_name:
                dataset = load_dataset(hf_name, config_name, split=split)
            else:
                dataset = load_dataset(hf_name, split=split)

            indices = sample_indices(len(dataset), num_items, seed)
            rows = [normalizer(dataset[i], dataset_name, n + 1) for n, i in enumerate(indices)]

            out_path = out_dir / f"{dataset_name}.jsonl"
            write_jsonl(out_path, rows)
            summary.append({"dataset": dataset_name, "n": len(rows), "split": split})
            print(f"  Wrote {len(rows)} records -> {out_path}")

        with (out_dir / "dataset_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)


    #2. Generate prompts

    MCQ_TEMPLATES = {
        "direct": "Answer the following multiple-choice question. Return only the option letter.\nQuestion: {question}\nOptions: {options}",
        "role": "You are a careful exam solver. Answer with only one option letter.\nQuestion: {question}\nOptions: {options}",
        "minimal": "{question}\n{options}\nLetter only:",
        "verbose": "Think carefully about the question, then return only the final option letter.\nQuestion: {question}\nOptions: {options}",
        "format_strict": "Choose the correct option and respond in the format 'Answer: X'.\nQuestion: {question}\nOptions: {options}",
        "bullet": "- Question: {question}\n- Choices: {options}\n- Response: one letter",
        "question_first": "Which option is correct?\n{question}\n{options}\nAnswer:",
        "narrative": "A student asks: {question}. The choices are {options}. Reply with the correct letter only.",
    }

    BOOLEAN_TEMPLATES = {
        "direct": "Answer yes or no only.\nQuestion: {question}\nContext: {context}",
        "role": "You are a precise verifier. Respond only 'yes' or 'no'.\nQuestion: {question}\nContext: {context}",
        "minimal": "Q: {question}\nContext: {context}\nAnswer yes/no:",
        "verbose": "Carefully inspect the context and answer the question with just yes or no.\nQuestion: {question}\nContext: {context}",
        "format_strict": "Return a single token: yes or no.\nQuestion: {question}\nContext: {context}",
        "bullet": "- Question: {question}\n- Context: {context}\n- Output: yes or no",
        "question_first": "Based on the context, is the answer yes?\nQuestion: {question}\nContext: {context}",
        "narrative": "A reader asks whether the statement is true: {question}. Use the context '{context}' and answer only yes or no.",
    }

    QA_TEMPLATES = {
        "direct": "Answer the question briefly using the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
        "role": "You are a concise question answering assistant. Respond with a short phrase.\nQuestion: {question}\nContext: {context}",
        "minimal": "Q: {question}\nContext: {context}\nA:",
        "verbose": "Read the context carefully and answer the question in a short span.\nQuestion: {question}\nContext: {context}",
        "format_strict": "Return only the answer span.\nQuestion: {question}\nContext: {context}",
        "bullet": "- Question: {question}\n- Context: {context}\n- Output: short answer only",
        "question_first": "What is the answer to: {question}?\nContext: {context}",
        "narrative": "Please answer this: {question}. Background: {context}. Give the direct answer.",
    }


    def build_prompts(dataset_name: str, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if dataset_name in {"arc_easy", "arc_challenge"}:
            templates = MCQ_TEMPLATES
        elif dataset_name == "boolq":
            templates = BOOLEAN_TEMPLATES
        else:
            templates = QA_TEMPLATES

        out = []
        for record in records:
            for template_name, template in templates.items():
                prompt = template.format(
                    question=record["question"],
                    options=" ".join(record.get("options", [])),
                    context=record.get("context", ""),
                )
                out.append({
                    "dataset": dataset_name,
                    "id": record["id"],
                    "task_type": record["task_type"],
                    "template_name": template_name,
                    "prompt": prompt,
                    "question": record["question"],
                    "context": record.get("context", ""),
                    "options": record.get("options", []),
                    "gold_answer": record["gold_answer"],
                    "gold_text": record.get("gold_text", record["gold_answer"]),
                })
        return out


    def generate_prompts(data_dir: Path, out_file: Path) -> None:
        datasets = ["arc_easy", "arc_challenge", "boolq", "squad"]
        all_prompts: List[Dict[str, Any]] = []
        for name in datasets:
            path = data_dir / f"{name}.jsonl"
            if not path.exists():
                print(f"  Skipping {name} (not found)")
                continue
            records = read_jsonl(path)
            prompts = build_prompts(name, records)
            all_prompts.extend(prompts)
        write_jsonl(out_file, all_prompts)
        print(f"  Wrote {len(all_prompts)} prompts -> {out_file}")


    #3. Run inference

    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text)


    def extract_choice_letter(text: str, options: Sequence[str]) -> str:
        upper = text.upper().strip()
        matches = re.findall(r"\b([A-Z])\b", upper)
        valid = {opt.split(")")[0] for opt in options}
        for m in matches:
            if m in valid:
                return m
        for opt in options:
            label, body = opt.split(")", 1)
            if normalize_text(body) and normalize_text(body) in normalize_text(text):
                return label.strip()
        return matches[0] if matches else upper[:1]


    def extract_yes_no(text: str) -> str:
        norm = normalize_text(text)
        if "yes" in norm:
            return "yes"
        if "no" in norm:
            return "no"
        return norm.split()[0] if norm else ""


    def token_f1(prediction: str, gold: str) -> float:
        pred_tok = normalize_text(prediction).split()
        gold_tok = normalize_text(gold).split()
        if not pred_tok or not gold_tok:
            return 0.0
        common = set(pred_tok) & set(gold_tok)
        if not common:
            return 0.0
        p = len(common) / len(pred_tok)
        r = len(common) / len(gold_tok)
        return 2 * p * r / (p + r)


    def evaluate_row(row: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
        task_type = row["task_type"]
        gold = row["gold_answer"]
        if task_type == "mcq":
            predicted = extract_choice_letter(raw_response, row.get("options", []))
            exact = normalize_text(predicted) == normalize_text(gold)
            f1 = None
        elif task_type == "boolean":
            predicted = extract_yes_no(raw_response)
            exact = normalize_text(predicted) == normalize_text(gold)
            f1 = None
        else:
            predicted = raw_response.strip()
            exact = normalize_text(predicted) == normalize_text(gold)
            f1 = token_f1(predicted, gold)
        return {"predicted": predicted, "exact_match": bool(exact), "f1": f1}


    def infer_ollama(prompt: str, model: str, temperature: float, max_tokens: int,
                     base_url: str, api_key: str | None = None) -> str:
        import requests
        base = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a careful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(base + "/api/chat", json=payload, headers=headers, timeout=180)
        if response.status_code == 404:
            gen_payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            response = requests.post(base + "/api/generate", json=gen_payload, headers=headers, timeout=180)
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


    def infer_openai(prompt: str, model: str, temperature: float, max_tokens: int,
                     base_url: str | None = None, api_key: str | None = None) -> str:
        from openai import OpenAI
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and not base_url:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        # For local OpenAI-compatible servers (LiteLLM, vLLM, Ollama /v1) a
        # placeholder key is required by the SDK even if the server ignores it.
        if not resolved_key:
            resolved_key = "placeholder"
        kwargs = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are a careful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


    def infer_hf_api(prompt: str, model: str, temperature: float, max_tokens: int, endpoint: str | None = None) -> str:
        import requests
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        headers = {"Content-Type": "application/json"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        urls = [endpoint] if endpoint else []
        if not endpoint:
            if hf_token:
                urls.append(f"https://router.huggingface.co/hf-inference/models/{model}")
            urls.append(f"https://api-inference.huggingface.co/models/{model}")

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False},
            "options": {"wait_for_model": True},
        }

        for url in urls:
            r = requests.post(url, json=payload, headers=headers, timeout=240)
            if r.status_code in (400, 404, 422, 503) and not endpoint:
                continue
            r.raise_for_status()
            obj = r.json()
            if isinstance(obj, list) and obj:
                return str(obj[0].get("generated_text", "")).strip()
            if isinstance(obj, dict) and "generated_text" in obj:
                return str(obj["generated_text"]).strip()
            raise RuntimeError(f"Unexpected HF response: {obj}")
        raise RuntimeError(f"All HF endpoints failed for model: {model}")


    def run_inference(in_file: Path, out_file: Path, backend: str, model: str,
                      temperature: float, max_tokens: int, limit: int,
                      ollama_base_url: str, ollama_api_key: str | None,
                      openai_compat_url: str | None, openai_compat_key: str | None,
                      hf_api_endpoint: str | None) -> None:
        rows = read_jsonl(in_file)
        if limit > 0:
            rows = rows[:limit]

        print(f"  Backend: {backend}, Model: {model}, Items: {len(rows)}")

        outputs = []
        hf_cache: Dict[str, Any] = {}

        for i, row in enumerate(rows, 1):
            prompt = row["prompt"]

            if backend == "ollama":
                raw = infer_ollama(prompt, model, temperature, max_tokens, ollama_base_url, ollama_api_key)
            elif backend == "openai":
                raw = infer_openai(prompt, model, temperature, max_tokens)
            elif backend == "openai_compat":
                raw = infer_openai(prompt, model, temperature, max_tokens,
                                   base_url=openai_compat_url, api_key=openai_compat_key)
            elif backend == "hf_api":
                raw = infer_hf_api(prompt, model, temperature, max_tokens, hf_api_endpoint)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            evaluation = evaluate_row(row, raw)
            outputs.append({**row, "item_id": row["id"], "backend": backend, "model": model,
                             "raw_response": raw, "response": raw, **evaluation})

            if i % 25 == 0:
                print(f"  Processed {i}/{len(rows)}")

        write_jsonl(out_file, outputs)
        print(f"  Wrote {len(outputs)} responses -> {out_file}")


    #4. Summarize

    def summarize(in_file: Path, out_dir: Path) -> None:
        import numpy as np
        import pandas as pd

        rows = read_jsonl(in_file)
        if not rows:
            raise RuntimeError("No rows found.")

        df = pd.DataFrame(rows)
        df["exact_match"] = df["exact_match"].astype(int)
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

        raw_csv = out_dir / "inference_results.csv"
        df.to_csv(raw_csv, index=False)

        template_summary = (
            df.groupby(["dataset", "template_name"])
            .agg(n=("exact_match", "size"), exact_match=("exact_match", "mean"),
                 f1=("f1", lambda x: float(x.fillna(0).mean())))
            .reset_index()
        )
        template_summary.to_csv(out_dir / "template_summary.csv", index=False)

        item_summary = (
            df.groupby(["dataset", "id"])
            .agg(exact_match_std=("exact_match", lambda x: float(np.std(x, ddof=0))),
                 exact_match_mean=("exact_match", "mean"))
            .reset_index()
        )
        item_summary.to_csv(out_dir / "item_sensitivity.csv", index=False)

        sens = item_summary.groupby("dataset")["exact_match_std"].mean().reset_index()
        sens.columns = ["dataset", "mean_item_std"]
        sens.to_csv(out_dir / "dataset_sensitivity.csv", index=False)

        print(f"  Saved results to {out_dir}/")
        print(template_summary.to_string(index=False))


    #5. Metrics

    def compute_metrics(in_file: Path, out_dir: Path, figures_dir: Path,
                        use_embeddings: bool = True) -> None:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(in_file) if in_file.suffix == ".csv" else pd.DataFrame(read_jsonl(in_file))

        if "item_id" not in df.columns and "id" in df.columns:
            df["item_id"] = df["id"]
        if "is_correct" not in df.columns and "exact_match" in df.columns:
            df["is_correct"] = df["exact_match"]
        if "backend" not in df.columns:
            df["backend"] = "unknown"

        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
        df["response"] = df["response"].fillna("").astype(str)

        sim_fn = _get_sim_fn(use_embeddings)

        rows = []
        for keys, group in df.groupby(["dataset", "item_id", "model", "backend"], dropna=False):
            dataset, item_id, model, backend = keys
            acc = group["is_correct"].astype(float).to_numpy()
            sens_acc = float(np.std(acc, ddof=0))
            mean_acc = float(np.mean(acc))

            responses = group["response"].tolist()
            sim_matrix = sim_fn(responses)
            n = sim_matrix.shape[0]
            if n >= 2:
                idx = np.triu_indices(n, k=1)
                pair_vals = sim_matrix[idx]
                stab_sem = float(np.mean(pair_vals))
            else:
                stab_sem = 1.0

            eval_artifact = sens_acc * stab_sem
            real_instability = sens_acc * (1.0 - stab_sem)

            if sens_acc < 0.15:
                category = "STABLE_OR_LOW_VARIANCE"
            elif stab_sem >= 0.75:
                category = "LIKELY_EVAL_ARTIFACT"
            elif stab_sem <= 0.45:
                category = "LIKELY_REAL_INSTABILITY"
            else:
                category = "MIXED"

            rows.append({"dataset": dataset, "item_id": item_id, "model": model, "backend": backend,
                         "SensAcc": sens_acc, "StabSem": stab_sem, "mean_accuracy": mean_acc,
                         "EvalArtifactScore": eval_artifact, "RealInstabilityScore": real_instability,
                         "Category": category})

        item_df = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        item_df.to_csv(out_dir / "metrics_summary.csv", index=False)
        print("\nCategory distribution:")
        print(item_df["Category"].value_counts())
        corr = float(item_df["SensAcc"].corr(item_df["StabSem"]))
        print(f"Correlation(SensAcc, StabSem): {corr:.4f}")

        _make_plots(item_df, figures_dir)


    def _get_sim_fn(prefer_embeddings: bool):
        def token_jaccard(texts):
            import numpy as np
            n = len(texts)
            mat = np.zeros((n, n))
            sets = [set(str(t).lower().split()) for t in texts]
            for i in range(n):
                mat[i, i] = 1.0
                for j in range(i + 1, n):
                    u = sets[i] | sets[j]
                    s = len(sets[i] & sets[j]) / len(u) if u else 1.0
                    mat[i, j] = mat[j, i] = s
            return mat

        if prefer_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                _model = SentenceTransformer("all-MiniLM-L6-v2")

                def embedding_sim(texts):
                    emb = _model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-8, norms)
                    normed = emb / norms
                    sims = np.clip((normed @ normed.T + 1.0) / 2.0, 0.0, 1.0)
                    return sims

                print("  Using sentence-transformer embeddings for semantic similarity.")
                return embedding_sim
            except ImportError:
                pass

        print("  sentence-transformers not found; falling back to token Jaccard similarity.")
        return token_jaccard


    def _make_plots(item_df, figures_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plots.")
            return

        figures_dir.mkdir(parents=True, exist_ok=True)
        colors = {
            "STABLE_OR_LOW_VARIANCE": "#2ecc71",
            "LIKELY_EVAL_ARTIFACT": "#f39c12",
            "LIKELY_REAL_INSTABILITY": "#e74c3c",
            "MIXED": "#3498db",
        }

        fig, ax = plt.subplots(figsize=(9, 6))
        for cat in item_df["Category"].unique():
            sub = item_df[item_df["Category"] == cat]
            ax.scatter(sub["SensAcc"], sub["StabSem"], alpha=0.65, s=60, label=cat,
                       color=colors.get(cat, "#999"))
        ax.axhline(0.75, linestyle="--", color="gray", alpha=0.5)
        ax.axhline(0.45, linestyle=":", color="gray", alpha=0.5)
        ax.axvline(0.15, linestyle="--", color="gray", alpha=0.5)
        ax.set_xlabel("Accuracy Sensitivity (SensAcc)")
        ax.set_ylabel("Semantic Stability (StabSem)")
        ax.set_title(f"Item-Level Sensitivity vs Stability (n={len(item_df)})")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "sensitivity_vs_stability.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        counts = item_df["Category"].value_counts()
        ax.bar(range(len(counts)), counts.values, color=[colors.get(c, "#999") for c in counts.index],
               alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=15, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Items by Stability Category")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(figures_dir / "category_distribution.png", dpi=180)
        plt.close(fig)

        print(f"  Saved plots -> {figures_dir}/")


    # I/O utilities


    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


    def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")


    # Main 

    def main() -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv("att1/.env")
        except ImportError:
            pass

        parser = argparse.ArgumentParser(description="Full prompt-sensitivity pipeline (no Marimo required).")
        parser.add_argument("--backend", choices=["ollama", "openai", "openai_compat", "hf_api"],
                            default=os.getenv("INFER_BACKEND", "ollama"))
        parser.add_argument("--model", default=os.getenv("INFER_MODEL", "llama3.1:8b"))
        parser.add_argument("--num-items", type=int, default=15)
        parser.add_argument("--limit", type=int, default=0, help="Cap total prompt count (0 = no cap)")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=64)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        parser.add_argument("--ollama-api-key", default=os.getenv("OLLAMA_API_KEY"),
                            help="Bearer token for authenticated Ollama /api/* deployments.")
        parser.add_argument("--openai-compat-url", default=os.getenv("OPENAI_COMPAT_URL"),
                            help="Base URL for OpenAI-compatible servers (LiteLLM, vLLM, Ollama /v1). "
                                 "Example: http://localhost:2720/v1")
        parser.add_argument("--openai-compat-key", default=os.getenv("OPENAI_COMPAT_KEY"),
                            help="API key for the OpenAI-compatible server (leave empty if not required).")
        parser.add_argument("--hf-api-endpoint", default=None)
        parser.add_argument("--data-dir", default="att1/data")
        parser.add_argument("--output-dir", default="att1/output")
        parser.add_argument("--skip-data-prep", action="store_true",
                            help="Skip dataset download (use existing att1/data/*.jsonl)")
        parser.add_argument("--skip-inference", action="store_true",
                            help="Skip inference (use existing responses.jsonl)")
        parser.add_argument("--no-embeddings", action="store_true",
                            help="Use token Jaccard instead of sentence-transformer embeddings")
        args = parser.parse_args()

        random.seed(args.seed)
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        prompts_file = output_dir / "prompts.jsonl"
        responses_file = output_dir / "responses.jsonl"
        raw_results_dir = Path("att1/data/raw_results")
        figures_dir = Path("att1/analysis/figures")

        if not args.skip_data_prep:
            print("\n[Step 1] Downloading & normalizing datasets...")
            prepare_data(data_dir, args.num_items, args.seed)

        print("\n[Step 2] Generating prompt variants...")
        generate_prompts(data_dir, prompts_file)

        if not args.skip_inference:
            print(f"\n[Step 3] Running inference ({args.backend} / {args.model})...")
            run_inference(
                in_file=prompts_file,
                out_file=responses_file,
                backend=args.backend,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                limit=args.limit,
                ollama_base_url=args.ollama_base_url,
                ollama_api_key=args.ollama_api_key,
                openai_compat_url=args.openai_compat_url,
                openai_compat_key=args.openai_compat_key,
                hf_api_endpoint=args.hf_api_endpoint,
            )

        print("\n[Step 4] Summarizing results...")
        summarize(responses_file, raw_results_dir)

        print("\n[Step 5] Computing sensitivity & artifact metrics...")
        compute_metrics(
            in_file=raw_results_dir / "inference_results.csv",
            out_dir=raw_results_dir,
            figures_dir=figures_dir,
            use_embeddings=not args.no_embeddings,
        )

        print("\nPipeline complete.")


    if __name__ == "__main__":
        main()

    return


if __name__ == "__main__":
    app.run()
