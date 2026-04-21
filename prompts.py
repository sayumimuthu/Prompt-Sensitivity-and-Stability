import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():

    #Generate Prompt Variants & Run Initial Inferences

    import json
    import csv
    from typing import List, Dict
    import os


    return Dict, List, csv, json, os


@app.cell
def _(json, os):
    # Create project directory structure
    os.makedirs("data/datasets", exist_ok=True)
    os.makedirs("data/prompts", exist_ok=True)
    os.makedirs("data/raw_results", exist_ok=True)
    os.makedirs("src", exist_ok=True)

    print("Directory structure created")

    #Load Sample Datasets (or create minimal versions)
    #For QUICK START: Using tiny samples

    SAMPLE_DATA = {
        "mmlu": [
            {
                "id": "mmlu_1",
                "question": "What is the capital of France?",
                "options": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
                "correct_answer": "B"
            },
            {
                "id": "mmlu_2",
                "question": "What is 2+2?",
                "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
                "correct_answer": "B"
            }
        ],
        "squad": [
            {
                "id": "squad_1",
                "question": "What is the smallest country in the world?",
                "context": "Vatican City is the smallest country in the world.",
                "gold_answer": "Vatican City"
            },
            {
                "id": "squad_2",
                "question": "What color is the sky?",
                "context": "The sky appears blue due to Rayleigh scattering.",
                "gold_answer": "blue"
            }
        ]
    }

    # Save sample datasets
    with open("data/datasets/sample_mmlu.json", "w") as f:
        json.dump(SAMPLE_DATA["mmlu"], f, indent=2)

    with open("data/datasets/sample_squad.json", "w") as f:
        json.dump(SAMPLE_DATA["squad"], f, indent=2)

    print("Sample datasets created (data/datasets/)")
    return (SAMPLE_DATA,)


@app.cell
def _(Dict, json):
    #Define prompt variants

    def create_prompt_variants() -> Dict[str, Dict[str, str]]:
        """
        Define 8 template variants for each task type
        """
        variants = {
            "mmlu": {
                "template_A_direct": "Answer the following multiple choice question. Question: {question} Options: {options} Answer:",
                "template_B_role": "As an expert, answer this multiple choice question. Question: {question} Options: {options} Answer:",
                "template_C_verbose": "Please provide a detailed answer to this multiple choice question. Question: {question} Options: {options} Provide only the letter.",
                "template_D_minimal": "{question} {options} Answer:",
                "template_E_numbered": "(1) Question: {question} (2) Options: {options} (3) Answer:",
                "template_F_bullets": "• Question: {question}\n• Options: {options}\n• Provide your answer:",
                "template_G_question": "Which of the following is correct? {question} {options}",
                "template_H_narrative": "{question} Among {options}, the correct answer is:",
            },
            "squad": {
                "template_A_direct": "Read the context and answer the question. Context: {context} Question: {question} Answer:",
                "template_B_role": "As a reading comprehension expert: Context: {context} Question: {question} Answer:",
                "template_C_verbose": "Based on the provided context, please answer the following question. Context: {context} Question: {question} Your answer:",
                "template_D_minimal": "{context} Q: {question} A:",
                "template_E_numbered": "(1) Context: {context} (2) Question: {question} (3) Answer:",
                "template_F_bullets": "• Context: {context}\n• Question: {question}\n• Answer the question directly:",
                "template_G_question": "From the given context, answer: {question} Context: {context}",
                "template_H_narrative": "Reading this context: {context}. Now, what is the answer to '{question}'?",
            }
        }
        return variants

    variants = create_prompt_variants()

    # Save variants to file for reference
    with open("data/prompts/variants.json", "w") as _f:
        json.dump(variants, _f, indent=2)

    print(" Prompt variants defined (data/prompts/variants.json)")
    print(f"   Templates per task: {len(variants['mmlu'])} variants")
    return (variants,)


@app.cell
def _(Dict, List, SAMPLE_DATA, variants):
    #Generate prompts (Ready for models)

    def format_prompts_for_inference(task_type: str, data: List[Dict]) -> List[Dict]:
        """
        Create all prompt variants for each item
        Returns: List of {item_id, template_name, formatted_prompt}
        """
        results = []
        task_variants = variants[task_type]

        for item in data:
            for template_name, template in task_variants.items():
                if task_type == "mmlu":
                    formatted = template.format(
                        question=item["question"],
                        options=" ".join(item["options"])
                    )
                else:  # squad
                    formatted = template.format(
                        context=item["context"],
                        question=item["question"]
                    )

                results.append({
                    "item_id": item["id"],
                    "template_name": template_name,
                    "full_prompt": formatted,
                    "correct_answer": item.get("correct_answer", item.get("gold_answer"))
                })

        return results

    mmlu_prompts = format_prompts_for_inference("mmlu", SAMPLE_DATA["mmlu"])
    squad_prompts = format_prompts_for_inference("squad", SAMPLE_DATA["squad"])

    print(f"Generated {len(mmlu_prompts)} MMLU prompts")
    print(f"Generated {len(squad_prompts)} SQuAD prompts")
    return mmlu_prompts, squad_prompts


@app.cell
def _(Dict, List, mmlu_prompts, squad_prompts):
    #Mock inference results (Replace with real API calls)

    def mock_inference(prompt: str, model_name: str) -> str:
        """
        REPLACE THIS with actual model calls
        For now, returns mock responses to test the pipeline
        """
        mock_responses = {
            "llama2": ["B", "Paris", "Vatican City"],
            "flan_t5": ["B", "Paris", "Vatican"],
            "gpt4_mini": ["B", "Paris", "Vatican City"],
        }
        import random
        return random.choice(mock_responses.get(model_name, ["A"]))

    def run_inference_batch(prompts: List[Dict], model_name: str) -> List[Dict]:
        """
        Run inference on all prompts for a given model
        """
        results = []
        for p in prompts:
            response = mock_inference(p["full_prompt"], model_name)  # Replace with real API
            results.append({
                "item_id": p["item_id"],
                "template_name": p["template_name"],
                "model": model_name,
                "response": response,
                "gold_answer": p["correct_answer"],
                "is_correct": response.strip().upper() == p["correct_answer"].upper()
            })
        return results

    #Run mock inference on sample data

    print("\n Running inference...")
    all_results = []

    for model in ["llama2", "flan_t5", "gpt4_mini"]:
        mmlu_results = run_inference_batch(mmlu_prompts, model)
        squad_results = run_inference_batch(squad_prompts, model)
        all_results.extend(mmlu_results + squad_results)

    print(f" Completed inference ({len(all_results)} prompts)")
    return (all_results,)


@app.cell
def _(all_results, csv):
    #Save results to CSV

    with open("data/raw_results/inference_results.csv", "w", newline="") as __f:
        writer = csv.DictWriter(__f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(" Results saved to data/raw_results/inference_results.csv")

    # ============================================================================
    # STEP 7: Basic Analysis
    # ============================================================================

    # Calculate accuracy per model per template
    accuracy_summary = {}
    for result in all_results:
        key = f"{result['model']}_{result['template_name']}"
        if key not in accuracy_summary:
            accuracy_summary[key] = {"total": 0, "correct": 0}

        accuracy_summary[key]["total"] += 1
        accuracy_summary[key]["correct"] += int(result["is_correct"])

    print("\n ACCURACY BY MODEL & TEMPLATE:")
    print("-" * 60)
    for key in sorted(accuracy_summary.keys()):
        total = accuracy_summary[key]["total"]
        correct = accuracy_summary[key]["correct"]
        acc = correct / total if total > 0 else 0
        print(f"{key:40} {acc*100:5.1f}% ({correct}/{total})")

    print("\n-" * 60)
    print(" Week 1 Quick Start Complete!")
    print("\n NEXT STEPS:")
    print("1. Replace mock_inference() with real API calls (OpenAI, Ollama, etc.)")
    print("2. Expand sample datasets to full size (15-20 items per task)")
    print("3. Run with all 3 models")
    print("4. In Week 2, compute semantic stability metrics")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
