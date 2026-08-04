#!/bin/bash
#SBATCH --job-name=faithshop-annotate
#SBATCH --output=results/logs/faith-%j.out
#SBATCH --error=results/logs/faith-%j.out
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=6:00:00
#SBATCH -C inet

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${LLM_PORT:-8000}"
# how many final sentences from the solver trace to include in the prompt
LAST_N_SENTENCES="${LAST_N_SENTENCES:-100}"

TP_SIZE="${SLURM_GPUS_ON_NODE:-4}"            # tensor-parallel size (GPUs on node)
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"

mkdir -p results
mkdir -p results/logs

module load miniforge3 gcc/13.2.0 cuda/12.6.2 apptainer

LOG_FILE="results/logs/vllm_annotate_${SLURM_JOB_ID:-$$}.log"
echo "Starting vLLM server (model=$MODEL) -> $LOG_FILE"
apptainer exec -B "$HF_HOME":"$HF_HOME":rw --nv --cleanenv ~/vllm.sif vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --trust-remote-code \
  --download-dir "$HF_HOME" \
  --max-model-len 10000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server readiness
for _ in {1..3600}; do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "vLLM ready"
    break
  fi
  printf "."; sleep 1
done

echo
echo "Running annotator with model=$MODEL port=$PORT"

# module load or otherwise set up environment
source activate fact-llm-env
cd src
# Process
python faith_shop.py --model "$MODEL" --port "$PORT"

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
echo "Output directory: $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi

