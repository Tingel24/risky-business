#!/bin/bash
#SBATCH --job-name=faith-qwq
#SBATCH --output=results/logs/faith-%j.out
#SBATCH --error=results/logs/faith-%j.out
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=10:00:00

set -euo pipefail

HF_HOME=$(ws_find models)
export HF_HOME

MODEL="${MODEL:-Qwen/QwQ-32B}"
# Automatically find an open ephemeral port
RANDOM_PORT=$(python -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
')

# Check if HF_HOME is actually set
if [ -z "$HF_HOME" ]; then
    echo "Error: HF_HOME environment variable is not set."
    exit 1
fi

# 1. Format the model folder name (replace '/' with '--')
FORMATTED_NAME="models--${MODEL//\//--}"

# 2. Define the path to the model's main cache directory
REPO_PATH="$HF_HOME/hub/$FORMATTED_NAME"

# 3. Read the active commit hash
# Hugging Face stores the current commit hash inside the 'refs/main' file
if [ -f "$REPO_PATH/refs/main" ]; then
    COMMIT_HASH=$(cat "$REPO_PATH/refs/main")
else
    # Fallback: Just grab the most recently modified directory inside snapshots
    COMMIT_HASH=$(ls -t "$REPO_PATH/snapshots/" 2>/dev/null | head -1)
fi

# 4. Assemble the full, final path
FINAL_PATH="$REPO_PATH/snapshots/$COMMIT_HASH"

# 5. Verify config.json exists and output the path
if [ -f "$FINAL_PATH/config.json" ]; then
    echo "Target directory located:"
    echo "$FINAL_PATH"

else
    echo "Warning: Path assembled, but config.json is missing:"
    echo "$FINAL_PATH"
fi

PORT="${LLM_PORT:-$RANDOM_PORT}"
# how many final sentences from the solver trace to include in the prompt
LAST_N_SENTENCES="${LAST_N_SENTENCES:-100}"

TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"            # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"

mkdir -p results
mkdir -p results/logs

module load miniforge3 gcc/13.2.0 cuda/12.6.2 apptainer

LOG_FILE="results/logs/vllm_annotate_${SLURM_JOB_ID:-$$}.log"
echo "Starting vLLM server (model=$MODEL) -> $LOG_FILE"
apptainer exec -B "$HF_HOME":"$HF_HOME":rw --nv --cleanenv ~/vllm.sif vllm serve "$FINAL_PATH" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --download-dir "$HF_HOME" \
  --max-model-len 10000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server readiness
echo "Waiting for vLLM to spin up and load weights..."
for _ in {1..1000}; do
  # -f makes curl fail on HTTP errors (like 503); /health only returns 200 OK when fully ready
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "vLLM ready!"
    break
  fi
  printf "."; sleep 2
done

echo
echo "Running annotator with model=$MODEL port=$PORT"

# module load or otherwise set up environment
source activate fact-llm-env-eval
cd src
# Process
python faith_shop_test_set.py --model "$MODEL" --port "$PORT"

# cleanup: politely stop server and wait a short time
if kill -0 $SERVER_PID 2>/dev/null; then
  echo "Stopping vLLM server..."
  kill -SIGINT $SERVER_PID || true
  sleep 2
  wait $SERVER_PID 2>/dev/null || true
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ANNOTATION BATCH COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Added fallback to prevent set -u from crashing if OUTPUT_DIR is undefined
echo "Output directory: ${OUTPUT_DIR:-Not Specified}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Added fallback to prevent set -u from crashing if FAIL_COUNT is undefined
if [[ ${FAIL_COUNT:-0} -gt 0 ]]; then
  exit 1
fi
