"""
LLM-as-Judge: for each response, ask a judge LLM whether it correctly answers
the question. Adds 'judge_correct' (bool) and 'judge_raw' (str) to each row.

The judge prompt follows Hua et al. (2509.01790v1) and is validated against
human annotations. We use the same Ollama model as inference by default
(llama3.1:8b), which correlates well with GPT-4 judgments (rank corr ~0.96).

Usage:
    python study/judge.py \
        --in-file  study/output/llama_responses.jsonl \
        --out-file study/output/llama_judged.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator. "
    "Respond with only the single word 'correct' or 'incorrect'. "
    "Do not add any explanation."
)

JUDGE_TEMPLATE = """\
You are evaluating whether a model's answer correctly answers a question.

Question: {question}
Reference answer: {gold_answer}
Model's answer: {model_answer}

Does the model's answer correctly answer the question?
Ignore minor formatting differences, capitalization, and paraphrasing — focus on whether the meaning is correct.
Respond with only "correct" or "incorrect".\
"""


def _load_env(path: str) -> None:
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def call_judge_ollama(judge_prompt: str, model: str,
                      base_url: str, api_key: str | None) -> str:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
        "options": {"temperature": 0.0, "num_predict": 10},
    }

    for endpoint in ("/api/chat", "/api/generate"):
        url = base_url.rstrip("/") + endpoint
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            obj = resp.json()
            if endpoint == "/api/chat":
                return obj.get("message", {}).get("content", "").strip().lower()
            else:
                return obj.get("response", "").strip().lower()
        except requests.HTTPError:
            if resp.status_code == 404:
                continue
            raise

    raise RuntimeError(f"Judge call failed: model={model} at {base_url}")


def parse_verdict(text: str) -> bool:
    """Return True if judge said 'correct', False otherwise. Conservative default: False."""
    t = text.lower().strip()
    # 'incorrect' must be checked first since it contains 'correct'
    if "incorrect" in t:
        return False
    if "correct" in t:
        return True
    return False


def main() -> None:
    _load_env("att1/.env")
    _load_env("study/.env")

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation pass.")
    parser.add_argument("--in-file",         default="study/output/responses.jsonl")
    parser.add_argument("--out-file",        default="study/output/judged.jsonl")
    parser.add_argument("--judge-model",     default=os.getenv("INFER_MODEL", "llama3.1:8b"))
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-api-key",  default=os.getenv("OLLAMA_API_KEY"))
    parser.add_argument("--limit",           type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.in_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"LLM-as-Judge: model={args.judge_model}  n={len(rows)}")

    outputs: List[Dict[str, Any]] = []
    errors = 0
    correct_count = 0

    for i, row in enumerate(rows, 1):
        question = row.get("question", "")
        gold     = row.get("gold_answer", "")
        response = row.get("raw_response", "") or "(no response)"

        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            gold_answer=gold,
            model_answer=response,
        )

        try:
            judge_raw     = call_judge_ollama(judge_prompt, args.judge_model,
                                               args.ollama_base_url, args.ollama_api_key)
            judge_correct = parse_verdict(judge_raw)
        except Exception as exc:
            print(f"  JUDGE ERROR row {i}: {exc}")
            judge_raw     = "error"
            judge_correct = False
            errors += 1

        if judge_correct:
            correct_count += 1

        outputs.append({**row, "judge_raw": judge_raw, "judge_correct": judge_correct})

        if i % 50 == 0 or i == len(rows):
            pct = correct_count / i * 100
            print(f"  {i}/{len(rows)} judged  |  running accuracy: {pct:.1f}%  (errors: {errors})")

    write_jsonl(Path(args.out_file), outputs)
    print(f"Wrote {len(outputs)} rows to {args.out_file}")
    print(f"Overall judge accuracy: {correct_count}/{len(outputs)} = {correct_count/len(outputs)*100:.1f}%")


if __name__ == "__main__":
    main()
