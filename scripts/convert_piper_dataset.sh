#!/bin/bash
set -e

# Usage: ./convert_piper_dataset.sh [dataset_name] [max_episodes]
# Example: ./convert_piper_dataset.sh pick_banana_50 10
# Example: ./convert_piper_dataset.sh pick_banana_50 -1 (for all)

DATASET_NAME=${1:-"pick_banana_50"}
MAX_EPISODES=${2:--1} # Default to -1 (all) if not provided

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Define the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Define the builder directory
BUILDER_DIR="$SCRIPT_DIR"

echo "================================================="
echo "RLDS Dataset Converter for Piper Dataset"
echo "Dataset Config: $DATASET_NAME"
echo "Max Episodes: $MAX_EPISODES"
echo "================================================="

# Check for required Dependencies
if ! python3 -c "import tensorflow_datasets" &> /dev/null; then
    echo "Error: tensorflow-datasets is not installed."
    echo "pip install tensorflow tensorflow-datasets tensorflow-hub h5py"
    exit 1
fi

echo "Switching to builder directory: $BUILDER_DIR"
if [ ! -d "$BUILDER_DIR" ]; then
    echo "Error: Builder directory not found at $BUILDER_DIR"
    exit 1
fi
cd "$BUILDER_DIR"

# Define output directory
OUTPUT_DIR="$PROJECT_ROOT/datasets_rlds"
mkdir -p "$OUTPUT_DIR"

# Export env var for builder
export MAX_EPISODES=$MAX_EPISODES

# Default to CPU-only to avoid CUDA/PTX incompatibilities during conversion.
# Set PIPER_DISABLE_GPU=0 if you explicitly want to use GPU.
export PIPER_DISABLE_GPU=${PIPER_DISABLE_GPU:-1}

echo "Starting dataset build process..."
# We use tfds build command which wraps the python script
# We pass the config name to the builder
tfds build "$BUILDER_DIR/piper_dataset_dataset_builder.py" --overwrite \
    --config="$DATASET_NAME" \
    --data_dir="$OUTPUT_DIR"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "================================================="
    echo "Dataset conversion successful!"
    echo "You can find the generated tfrecords in $OUTPUT_DIR/$DATASET_NAME"
    echo "================================================="
else
    echo "Dataset conversion failed with error code $STATUS"
fi
