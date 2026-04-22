import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import json
    from pathlib import Path
    from typing import Any, Dict, List, Sequence

    return Any, Dict, List, Path, Sequence, argparse, json


@app.cell
def _():
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

    return BOOLEAN_TEMPLATES, MCQ_TEMPLATES, QA_TEMPLATES


@app.cell
def _(
    Any,
    BOOLEAN_TEMPLATES,
    Dict,
    List,
    MCQ_TEMPLATES,
    Path,
    QA_TEMPLATES,
    Sequence,
    json,
):
    def read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
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


    def format_options(options: Sequence[str]) -> str:
        return " ".join(options)


    def build_prompts(dataset_name: str, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if dataset_name in {"arc_easy", "arc_challenge"}:
            templates = MCQ_TEMPLATES
        elif dataset_name == "boolq":
            templates = BOOLEAN_TEMPLATES
        else:
            templates = QA_TEMPLATES

        out: List[Dict[str, Any]] = []
        for record in records:
            for template_name, template in templates.items():
                prompt = template.format(
                    question=record["question"],
                    options=format_options(record.get("options", [])),
                    context=record.get("context", ""),
                )
                out.append(
                    {
                        "dataset": dataset_name,
                        "id": record["id"],
                        "task_type": record["task_type"],
                        "template_name": template_name,
                        "prompt_family": template_name,
                        "prompt": prompt,
                        "question": record["question"],
                        "context": record.get("context", ""),
                        "options": record.get("options", []),
                        "gold_answer": record["gold_answer"],
                        "gold_text": record.get("gold_text", record["gold_answer"]),
                    }
                )
        return out


    return build_prompts, read_jsonl, write_jsonl


@app.cell
def _(Any, Dict, List, Path, argparse, build_prompts, read_jsonl, write_jsonl):
    def main() -> None:
        parser = argparse.ArgumentParser(description="Generate prompt variants for the Attempt 1 benchmark suite.")
        parser.add_argument("--data-dir", default="att1/data")
        parser.add_argument("--out-file", default="att1/output/prompts.jsonl")
        args = parser.parse_args()

        data_dir = Path(args.data_dir)
        datasets = ["arc_easy", "arc_challenge", "boolq", "squad"]
        all_prompts: List[Dict[str, Any]] = []

        for dataset_name in datasets:
            path = data_dir / f"{dataset_name}.jsonl"
            if not path.exists():
                continue
            records = read_jsonl(path)
            all_prompts.extend(build_prompts(dataset_name, records))

        out_file = Path(args.out_file)
        write_jsonl(out_file, all_prompts)
        print(f"Wrote {len(all_prompts)} prompts to {out_file}")


    if __name__ == "__main__":
        main()
    return


if __name__ == "__main__":
    app.run()
