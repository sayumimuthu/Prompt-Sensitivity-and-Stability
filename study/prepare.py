"""
2x2x2 prompt templates.

Three binary structural factors:
  R = Role framing    (0=absent, 1=present)
  F = Format directive(0=absent, 1=present)
  P = Answer prefix   (0=absent, 1=present)

Gives 8 templates per dataset (index = R*4 + F*2 + P).
Template ID string: T{R}{F}{P}, e.g. T101 = role + prefix, no format.
"""

from typing import Any, Dict, List


ROLE_TEXT = "You are a careful and accurate assistant.\n"

FORMAT_TEXT: Dict[str, str] = {
    "arc_challenge": "Respond with only the letter of the correct option (A, B, C, or D).\n",
    "boolq":         "Respond with only 'yes' or 'no'.\n",
    "squad":         "Respond with a short phrase or sentence directly answering the question.\n",
}

PREFIX_TEXT = "Answer:"


def make_template_id(role: bool, fmt: bool, prefix: bool) -> str:
    return f"T{int(role)}{int(fmt)}{int(prefix)}"


def build_prompt(item: Dict[str, Any], dataset: str,
                 role: bool, fmt: bool, prefix: bool) -> str:
    parts: List[str] = []

    if role:
        parts.append(ROLE_TEXT)

    if dataset == "arc_challenge":
        opts_text = "\n".join(item["options"])
        parts.append(f"Question: {item['question']}\n{opts_text}\n")
        if fmt:
            parts.append(FORMAT_TEXT["arc_challenge"])

    elif dataset == "boolq":
        parts.append(f"Passage: {item['passage']}\nQuestion: {item['question']}\n")
        if fmt:
            parts.append(FORMAT_TEXT["boolq"])

    elif dataset == "squad":
        parts.append(f"Context: {item['context']}\nQuestion: {item['question']}\n")
        if fmt:
            parts.append(FORMAT_TEXT["squad"])

    if prefix:
        parts.append(PREFIX_TEXT)

    return "".join(parts)


def all_templates(item: Dict[str, Any], dataset: str) -> List[Dict[str, Any]]:
    """Return list of 8 dicts: template_id, role, fmt, prefix, prompt."""
    results = []
    for role in (False, True):
        for fmt in (False, True):
            for prefix in (False, True):
                results.append({
                    "template_id": make_template_id(role, fmt, prefix),
                    "role": int(role),
                    "fmt": int(fmt),
                    "prefix": int(prefix),
                    "prompt": build_prompt(item, dataset, role, fmt, prefix),
                })
    return results


if __name__ == "__main__":
    # Quick check
    dummy = {
        "question": "What is 2+2?",
        "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
        "gold_answer": "B",
    }
    for t in all_templates(dummy, "arc_challenge"):
        print(f"\n {t['template_id']} (role={t['role']} fmt={t['fmt']} prefix={t['prefix']}) ---")
        print(t["prompt"])

