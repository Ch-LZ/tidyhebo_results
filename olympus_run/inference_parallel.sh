#!/bin/bash
# Run several optimizers across several Olympus tasks in parallel.
# Usage: ./inference_parallel.sh "optuna,random,hebo" "colors_bob,alkox,benzylation"

set -e

# Configuration
NUM_ITER=100
NUM_RUNS=30
OUTPUT_DIR="02_olympus"
LOG_LEVEL="INFO"
MAX_PARALLEL=2

# Arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 \"optimizers(comma-separated)\" \"tasks(comma-separated)\""
    echo "Example: $0 \"optuna,random,hebo\" \"colors_bob,alkox,benzylation\""
    exit 1
fi

OPTIMIZERS_STR="$1"
TASKS_STR="$2"

IFS=',' read -ra OPTIMIZERS <<< "$OPTIMIZERS_STR"
IFS=',' read -ra TASKS <<< "$TASKS_STR"

echo "Optimizers: ${OPTIMIZERS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Iterations: $NUM_ITER, runs: $NUM_RUNS"
echo "Results: $OUTPUT_DIR/"
echo "----------------------------------------"

JOBS=()
START_TIME=$(date +%s)

run_combination() {
    local opt=$1
    local task=$2
    echo "Starting: optimizer=$opt, task=$task"
    CMD="python orchestrator.py --dataset $task --optimizer $opt --num-iterations $NUM_ITER --num-runs $NUM_RUNS --output-dir $OUTPUT_DIR --log-level $LOG_LEVEL"
    eval $CMD
    if [ $? -eq 0 ]; then
        echo "Completed: $opt on $task"
    else
        echo "Failed: $opt on $task"
    fi
}

for opt in "${OPTIMIZERS[@]}"; do
    for task in "${TASKS[@]}"; do
        run_combination "$opt" "$task" &
        JOBS+=($!)
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done
    done
done

wait

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "----------------------------------------"
echo "All combinations completed in $ELAPSED seconds."
