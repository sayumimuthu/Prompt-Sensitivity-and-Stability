#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh  —  Run BOTH models, then combine for the cross-model comparison.
#
# Runs entirely via HF local (no Ollama, no API key required).
#   Model 1: meta-llama/Llama-3.1-8B-Instruct  (needs HF_TOKEN + ~16GB RAM)
#   Model 2: HuggingFaceTB/SmolLM2-1.7B-Instruct (no token, ~4GB RAM)
#   Judge  : HuggingFaceTB/SmolLM2-1.7B-Instruct (same small model)
#
# If you have Ollama running locally (no API key), set:
#   USE_OLLAMA=1 bash study/run_all.sh
#
# Usage:
#   bash study/run_all.sh           # full run (30 items per dataset)
#   N=10 bash study/run_all.sh      # quick smoke-test
# ---------------------------------------------------------------------------
set -euo pipefail

N="${N:-30}"
SEED="${SEED:-42}"
USE_OLLAMA="${USE_OLLAMA:-0}"

cd "$(dirname "$0")/.."

# Resolve Python: prefer myenv conda env, fall back to whatever is in PATH
MYENV_PY="/opt/anaconda3/envs/myenv/bin/python"
if [ -x "$MYENV_PY" ]; then
    PY="$MYENV_PY"
else
    PY="python"
    echo "WARNING: myenv not found at $MYENV_PY — using system python"
fi

echo "============================================================"
echo " Two-model prompt-sensitivity study"
echo " Python  : $PY"
echo " Items   : $N per dataset"
echo " Ollama  : $USE_OLLAMA (0=hf_local, 1=ollama)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0: Prepare shared prompts
# ---------------------------------------------------------------------------
echo ""
echo "[PREPARE] Building prompts..."
"$PY" study/prepare.py --n "$N" --seed "$SEED" --out-dir study/output
PROMPTS_FILE="study/output/prompts.jsonl"

# ---------------------------------------------------------------------------
# Model 1: LLaMA 3.1-8B-Instruct (HF local or Ollama)
# ---------------------------------------------------------------------------
echo ""
echo "=== MODEL 1: Llama-3.1-8B-Instruct ==="
SLUG1="Llama-3.1-8B-Instruct"
mkdir -p "study/output/${SLUG1}/figures"

if [ "$USE_OLLAMA" = "1" ]; then
    "$PY" study/infer.py \
        --in-file     "$PROMPTS_FILE" \
        --out-file    "study/output/${SLUG1}/responses.jsonl" \
        --backend     ollama \
        --model       llama3.1:8b \
        --temperature 0.0 \
        --max-tokens  64

    "$PY" study/judge.py \
        --in-file        "study/output/${SLUG1}/responses.jsonl" \
        --out-file       "study/output/${SLUG1}/judged.jsonl" \
        --judge-backend  ollama \
        --judge-model    llama3.1:8b
else
    "$PY" study/infer.py \
        --in-file     "$PROMPTS_FILE" \
        --out-file    "study/output/${SLUG1}/responses.jsonl" \
        --backend     hf_local \
        --model       meta-llama/Llama-3.1-8B-Instruct \
        --temperature 0.0 \
        --max-tokens  64

    "$PY" study/judge.py \
        --in-file        "study/output/${SLUG1}/responses.jsonl" \
        --out-file       "study/output/${SLUG1}/judged.jsonl" \
        --judge-backend  hf_local \
        --judge-model    HuggingFaceTB/SmolLM2-1.7B-Instruct
fi

# ---------------------------------------------------------------------------
# Model 2: SmolLM2-1.7B-Instruct (always HF local)
# ---------------------------------------------------------------------------
echo ""
echo "=== MODEL 2: SmolLM2-1.7B-Instruct ==="
SLUG2="SmolLM2-1.7B-Instruct"
mkdir -p "study/output/${SLUG2}/figures"

"$PY" study/infer.py \
    --in-file     "$PROMPTS_FILE" \
    --out-file    "study/output/${SLUG2}/responses.jsonl" \
    --backend     hf_local \
    --model       HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --temperature 0.0 \
    --max-tokens  64

"$PY" study/judge.py \
    --in-file        "study/output/${SLUG2}/responses.jsonl" \
    --out-file       "study/output/${SLUG2}/judged.jsonl" \
    --judge-backend  hf_local \
    --judge-model    HuggingFaceTB/SmolLM2-1.7B-Instruct

# ---------------------------------------------------------------------------
# Combined metrics and figures
# ---------------------------------------------------------------------------
echo ""
echo "=== COMBINED METRICS ==="
mkdir -p study/output/combined/figures

"$PY" study/metrics.py \
    --in-files \
        "study/output/${SLUG1}/judged.jsonl" \
        "study/output/${SLUG2}/judged.jsonl" \
    --out-instance study/output/combined/metrics_instance.csv \
    --out-dataset  study/output/combined/metrics_dataset.csv

"$PY" study/plot.py \
    --in-instance study/output/combined/metrics_instance.csv \
    --out-dir     study/output/combined/figures

echo ""
echo "All done. Results in study/output/combined/"
