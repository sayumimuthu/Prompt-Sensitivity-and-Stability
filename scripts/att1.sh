set -euo pipefail

# Usage examples:
# bash scripts/att1.sh
# BACKEND=openai MODEL=gpt-4o-mini bash scripts/att1.sh
# BACKEND=ollama MODEL=llama3.1:8b bash scripts/att1.sh
# BACKEND=hf MODEL=google/flan-t5-base bash scripts/att1.sh

BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-google/flan-t5-base}"
NUM_ITEMS="${NUM_ITEMS:-15}"
LIMIT="${LIMIT:-0}"
TEMP="${TEMP:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-200}"
SEED="${SEED:-42}"

python scripts/prepare_data.py \
  --mode hf \
  --num-items "$NUM_ITEMS" \
  --seed "$SEED" \
  --out-dir att1/data

python scripts/generate_prompts.py \
  --data-dir att1/data \
  --out-file att1/output/prompts.jsonl

python scripts/run_inference.py \
  --in-file att1/output/prompts.jsonl \
  --out-file att1/output/responses.jsonl \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --temperature "$TEMP" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --limit "$LIMIT"

python scripts/summarize_week1.py \
  --in-file att1/output/responses.jsonl \
  --out-csv att1/output/week1_summary.csv \
  --out-json att1/output/week1_summary.json

echo "Attempt 1 pipeline completed. Check att1/output/."


