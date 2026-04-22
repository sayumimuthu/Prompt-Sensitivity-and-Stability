import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import json
    import os
    import random
    import re
    from pathlib import Path
    from typing import Any, Dict, List, Sequence

    import requests
    from dotenv import load_dotenv

    return (
        Any,
        Dict,
        List,
        Path,
        Sequence,
        argparse,
        json,
        load_dotenv,
        os,
        random,
        re,
        requests,
    )


@app.cell
def _(Any, Dict, List, Path, Sequence, json, re):
    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


    def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")


    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


    def extract_choice_letter(text: str, options: Sequence[str]) -> str:
        upper = text.upper().strip()
        matches = re.findall(r"\b([A-Z])\b", upper)
        for match in matches:
            if match in {option.split(")")[0] for option in options}:
                return match

        for option in options:
            label, body = option.split(")", 1)
            if normalize_text(body) and normalize_text(body) in normalize_text(text):
                return label.strip()

        if matches:
            return matches[0]
        return upper[:1]


    def extract_yes_no(text: str) -> str:
        normalized = normalize_text(text)
        if "yes" in normalized:
            return "yes"
        if "no" in normalized:
            return "no"
        return normalized.split()[0] if normalized else ""


    def exact_match(prediction: str, gold: str) -> bool:
        return normalize_text(prediction) == normalize_text(gold)


    def token_f1(prediction: str, gold: str) -> float:
        pred_tokens = normalize_text(prediction).split()
        gold_tokens = normalize_text(gold).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)


    return (
        exact_match,
        extract_choice_letter,
        extract_yes_no,
        normalize_text,
        read_jsonl,
        token_f1,
        write_jsonl,
    )


@app.cell
def _(os):
    def infer_openai(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=api_key)
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


    return (infer_openai,)


@app.cell
def _(requests):
    def infer_ollama(prompt: str, model: str, temperature: float, max_tokens: int, base_url: str) -> str:
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a careful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        obj = response.json()
        return obj.get("message", {}).get("content", "").strip()

    return (infer_ollama,)


@app.cell
def _(Any, Dict):
    class HuggingFaceGenerator:
        def __init__(self, model_name: str):
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.kind = "seq2seq"
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.kind = "causal"

        def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                if self.kind == "seq2seq":
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=max(temperature, 1e-5),
                    )
                else:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=max(temperature, 1e-5),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return text.strip()


    def infer_hf(prompt: str, model_name: str, temperature: float, max_tokens: int, cache: Dict[str, Any]) -> str:
        if "generator" not in cache:
            cache["generator"] = HuggingFaceGenerator(model_name)
        return cache["generator"].generate(prompt, temperature, max_tokens)

    return (infer_hf,)


@app.cell
def _(
    Any,
    Dict,
    exact_match,
    extract_choice_letter,
    extract_yes_no,
    normalize_text,
    token_f1,
):
    def evaluate_row(row: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
        task_type = row["task_type"]
        gold = row["gold_answer"]
        predicted = raw_response
        exact = False
        f1 = None

        if task_type == "mcq":
            predicted = extract_choice_letter(raw_response, row.get("options", []))
            exact = normalize_text(predicted) == normalize_text(gold)
        elif task_type == "boolean":
            predicted = extract_yes_no(raw_response)
            exact = normalize_text(predicted) == normalize_text(gold)
        else:
            predicted = raw_response.strip()
            exact = exact_match(predicted, gold)
            f1 = token_f1(predicted, gold)

        return {
            "predicted": predicted,
            "exact_match": bool(exact),
            "f1": f1,
        }


    return (evaluate_row,)


@app.cell
def _(
    Any,
    Dict,
    List,
    Path,
    argparse,
    evaluate_row,
    infer_hf,
    infer_ollama,
    infer_openai,
    load_dotenv,
    os,
    random,
    read_jsonl,
    write_jsonl,
):
    def main() -> None:
        parser = argparse.ArgumentParser(description="Run prompt-sensitivity inference on real models.")
        parser.add_argument("--in-file", default="att1/output/prompts.jsonl")
        parser.add_argument("--out-file", default="att1/output/responses.jsonl")
        parser.add_argument("--backend", choices=["mock", "openai", "ollama", "hf"], default="mock")
        parser.add_argument("--model", default="gpt-4o-mini")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=64)
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--ollama-base-url", default=None)
        args = parser.parse_args()

        load_dotenv("att1/.env")
        random.seed(args.seed)

        rows = read_jsonl(Path(args.in_file))
        if args.limit > 0:
            rows = rows[: args.limit]

        cache: Dict[str, Any] = {}
        base_url = args.ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        outputs: List[Dict[str, Any]] = []

        for index, row in enumerate(rows, start=1):
            prompt = row["prompt"]

            if args.backend == "mock":
                raw_response = row["gold_answer"]
            elif args.backend == "openai":
                raw_response = infer_openai(prompt, args.model, args.temperature, args.max_tokens)
            elif args.backend == "ollama":
                raw_response = infer_ollama(prompt, args.model, args.temperature, args.max_tokens, base_url)
            else:
                raw_response = infer_hf(prompt, args.model, args.temperature, args.max_tokens, cache)

            evaluation = evaluate_row(row, raw_response)
            outputs.append(
                {
                    **row,
                    "item_id": row["id"],
                    "backend": args.backend,
                    "model": args.model,
                    "raw_response": raw_response,
                    "response": raw_response,
                    **evaluation,
                }
            )

            if index % 25 == 0:
                print(f"Processed {index}/{len(rows)} prompts")

        out_file = Path(args.out_file)
        write_jsonl(out_file, outputs)
        print(f"Wrote {len(outputs)} responses to {out_file}")


    if __name__ == "__main__":
        main()
    return


if __name__ == "__main__":
    app.run()
