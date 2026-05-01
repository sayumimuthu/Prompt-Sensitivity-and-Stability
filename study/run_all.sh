#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh  —  Run BOTH models, then combine for the cross-model comparison.
#
# Model 1: LLaMA 3.1:8B  via Ollama
# Model 2: SmolLM2-1.7B  via HF local (no API key needed)
#
# Usage:
#   bash study/run_all.sh
#   N=10 bash study/run_all.sh       # quick smoke-test with 10 items
# ---------------------------------------------------------------------------
set -euo pipefail

N="${N:-30}"
SEED="${SEED:-42}"

cd "$(dirname "$0")/.."


echo " Two-model prompt-sensitivity study"
echo " Items per dataset : $N"
echo " Seed              : $SEED"


# Prepare shared prompts once
PROMPTS_FILE="study/output/prompts.jsonl"
echo ""
echo "[PREPARE] Building prompts..."
python study/prepare.py --n "$N" --seed "$SEED" --out-dir study/output


# Model 1: LLaMA 3.1:8B via Ollama

echo ""
echo "=== MODEL 1: llama3.1:8b (ollama) ==="
SLUG1="llama3.1__8b"
mkdir -p "study/output/${SLUG1}/figures"

python study/infer.py \
    --in-file     "$PROMPTS_FILE" \
    --out-file    "study/output/${SLUG1}/responses.jsonl" \
    --backend     ollama \
    --model       llama3.1:8b \
    --temperature 0.0 \
    --max-tokens  64

python study/judge.py \
    --in-file     "study/output/${SLUG1}/responses.jsonl" \
    --out-file    "study/output/${SLUG1}/judged.jsonl" \
    --judge-model llama3.1:8b


# Model 2: SmolLM2-1.7B via HF local

echo ""
echo "=== MODEL 2: SmolLM2-1.7B-Instruct (hf_local) ==="
SLUG2="SmolLM2-1.7B-Instruct"
mkdir -p "study/output/${SLUG2}/figures"

python study/infer.py \
    --in-file     "$PROMPTS_FILE" \
    --out-file    "study/output/${SLUG2}/responses.jsonl" \
    --backend     hf_local \
    --model       HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --temperature 0.0 \
    --max-tokens  64

# Judge still uses Ollama (larger model as judge)
python study/judge.py \
    --in-file     "study/output/${SLUG2}/responses.jsonl" \
    --out-file    "study/output/${SLUG2}/judged.jsonl" \
    --judge-model llama3.1:8b


# Combined metrics and figures

echo ""
echo "COMBINED METRICS"
mkdir -p study/output/combined/figures

python study/metrics.py \
    --in-files \
        "study/output/${SLUG1}/judged.jsonl" \
        "study/output/${SLUG2}/judged.jsonl" \
    --out-instance study/output/combined/metrics_instance.csv \
    --out-dataset  study/output/combined/metrics_dataset.csv

python study/plot.py \
    --in-instance study/output/combined/metrics_instance.csv \
    --out-dir     study/output/combined/figures

echo ""
echo "All done. Combined results in study/output/combined/"


