MODEL="${MODEL:-Qwen/QwQ-32B}"
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
