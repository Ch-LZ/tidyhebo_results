#!/bin/bash
# Run several optimizers for one Olympus dataset.
# Usage: ./inference.sh "optuna,random,hebo" "colors_bob" [batch_size] [init_size] [adaptive_ceil]

set -e

# Configuration
NUM_ITER=100
NUM_RUNS=30
OUTPUT_DIR="../03_olympus"
LOG_LEVEL="DEBUG"

# Arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"optimizers(comma-separated)\" \"dataset\" [batch_size] [init_size] [adaptive_ceil]"
    echo "Example: $0 \"optuna,random,hebo\" \"colors_bob\" 4 2"
    echo "Adaptive example: $0 \"tidyhebo\" \"colors_bob\" 1 1 3"
    exit 1
fi

OPTIMIZERS_STR="$1"
DATASET="$2"
BATCH_SIZE="${3:-1}"
INIT_SIZE="${4:-1}"
ADAPTIVE_CEIL="${5:-}"

IFS=',' read -ra OPTIMIZERS <<< "$OPTIMIZERS_STR"

echo "Dataset: $DATASET"
echo "Optimizers: ${OPTIMIZERS[*]}"
echo "Iterations: $NUM_ITER, runs: $NUM_RUNS"
echo "Batch size: $BATCH_SIZE, initial size: $INIT_SIZE"
if [ -n "$ADAPTIVE_CEIL" ]; then
    echo "Adaptive cap: $ADAPTIVE_CEIL"
else
    echo "Adaptive cap: not set"
fi
echo "Results: $OUTPUT_DIR/$DATASET/"
echo "----------------------------------------"

START_TIME=$(date +%s)

for OPT in "${OPTIMIZERS[@]}"; do
    echo ""
    echo "========== Optimizer: $OPT =========="

    CMD="python orchestrator.py --dataset $DATASET --optimizer $OPT --num-iterations $NUM_ITER --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --init-size $INIT_SIZE --output-dir $OUTPUT_DIR --log-level $LOG_LEVEL"

    if [ -n "$ADAPTIVE_CEIL" ]; then
        CMD="$CMD --adaptive-ceil $ADAPTIVE_CEIL"
    fi

    echo "Running: $CMD"
    eval $CMD

    if [ $? -eq 0 ]; then
        echo "Optimizer $OPT completed successfully."
    else
        echo "Optimizer $OPT failed."
        exit 1
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "----------------------------------------"
echo "All optimizers completed in $ELAPSED seconds."
