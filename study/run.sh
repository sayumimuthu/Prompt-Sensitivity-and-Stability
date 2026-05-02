#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh  —  Single-model run for the prompt-sensitivity study.
#
# Usage examples:
#   bash study/run.sh                              # defaults (see below)
#   MODEL=llama3.1:8b  bash study/run.sh           # Ollama model
#   BACKEND=hf_local  MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct  bash study/run.sh
#   N=10 LIMIT=80 bash study/run.sh                # quick smoke-test
# ---------------------------------------------------------------------------
set -euo pipefail

BACKEND="${BACKEND:-hf_local}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
JUDGE_BACKEND="${JUDGE_BACKEND:-hf_local}"
JUDGE_MODEL="${JUDGE_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
N="${N:-30}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"       # 0 = no limit; set e.g. LIMIT=10 for a quick test

# Slugify the model name for file naming
MODEL_SLUG="${MODEL_SLUG:-$(echo "$MODEL" | tr '/:' '__')}"

OUT_DIR="study/output/${MODEL_SLUG}"

# Run from repo root regardless of where the script is called from
cd "$(dirname "$0")/.."

# Resolve Python: prefer myenv conda env, fall back to whatever is in PATH
MYENV_PY="/opt/anaconda3/envs/myenv/bin/python"
if [ -x "$MYENV_PY" ]; then
    PY="$MYENV_PY"
else
    PY="python"
    echo "WARNING: myenv not found at $MYENV_PY — using system python (packages may be missing)"
fi

echo " Model  : $MODEL"
echo " Backend: $BACKEND"
echo " N items: $N per dataset  (limit=$LIMIT)"
echo " Output : $OUT_DIR"


mkdir -p "$OUT_DIR/figures"

# Step 1: prepare prompts (shared across model runs)
PROMPTS_FILE="study/output/prompts.jsonl"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo ""
    echo "[1/4] Preparing prompts..."
    "$PY" study/prepare.py \
        --n    "$N" \
        --seed "$SEED" \
        --out-dir study/output
else
    echo "[1/4] Prompts already exist at $PROMPTS_FILE — skipping."
fi

# Step 2: inference
echo ""
echo "[2/4] Running inference..."
"$PY" study/infer.py \
    --in-file       "$PROMPTS_FILE" \
    --out-file      "${OUT_DIR}/responses.jsonl" \
    --backend       "$BACKEND" \
    --model         "$MODEL" \
    --temperature   0.0 \
    --max-tokens    64 \
    --limit         "$LIMIT"

# Step 3: LLM-as-Judge
echo ""
echo "[3/4] Running LLM-as-Judge..."
"$PY" study/judge.py \
    --in-file        "${OUT_DIR}/responses.jsonl" \
    --out-file       "${OUT_DIR}/judged.jsonl" \
    --judge-backend  "$JUDGE_BACKEND" \
    --judge-model    "$JUDGE_MODEL"

# Step 4: metrics (single model only; use run_all.sh for combined analysis)
echo ""
echo "[4/4] Computing metrics..."
"$PY" study/metrics.py \
    --in-files      "${OUT_DIR}/judged.jsonl" \
    --out-instance  "${OUT_DIR}/metrics_instance.csv" \
    --out-dataset   "${OUT_DIR}/metrics_dataset.csv"

"$PY" study/plot.py \
    --in-instance   "${OUT_DIR}/metrics_instance.csv" \
    --out-dir       "${OUT_DIR}/figures"

echo ""
echo "Done. Results in ${OUT_DIR}/"
