#!/bin/bash

# Coup RL Training Script
# Usage: ./scripts/run_training.sh [options]

set -e  # Exit on error

# Default values
CONFIG_FILE="configs/training_config.yaml"
DEVICE="cpu"
NUM_PLAYERS=2
TOTAL_TIMESTEPS=1000000
RESUME_CHECKPOINT=""
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-players)
      NUM_PLAYERS="$2"
      shift 2
      ;;
    --total-timesteps)
      TOTAL_TIMESTEPS="$2"
      shift 2
      ;;
    --resume)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --config FILE           Training configuration file (default: configs/training_config.yaml)"
      echo "  --device DEVICE         Device to use: cpu or cuda (default: cpu)"
      echo "  --num-players N         Number of players (default: 2)"
      echo "  --total-timesteps N     Total training timesteps (default: 1000000)"
      echo "  --resume CHECKPOINT     Resume from checkpoint file"
      echo "  --log-level LEVEL       Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
      echo "  --help                  Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                          # Basic CPU training"
      echo "  $0 --device cuda --num-players 4           # GPU training with 4 players"
      echo "  $0 --resume logs/checkpoints/checkpoint.pt # Resume from checkpoint"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Configuration file not found: $CONFIG_FILE"
  exit 1
fi

# Check if CUDA is available if requested
if [[ "$DEVICE" == "cuda" ]]; then
  if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    echo "Warning: CUDA requested but not available, falling back to CPU"
    DEVICE="cpu"
  fi
fi

# Create necessary directories
mkdir -p logs
mkdir -p logs/checkpoints

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Build command
CMD="python train/train_agent.py"
CMD="$CMD --config $CONFIG_FILE"
CMD="$CMD --device $DEVICE"
CMD="$CMD --num_players $NUM_PLAYERS"
CMD="$CMD --total_timesteps $TOTAL_TIMESTEPS"

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD="$CMD --resume $RESUME_CHECKPOINT"
fi

# Print configuration
echo "=========================================="
echo "Coup RL Training Configuration"
echo "=========================================="
echo "Config file:     $CONFIG_FILE"
echo "Device:          $DEVICE"
echo "Players:         $NUM_PLAYERS"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Log level:       $LOG_LEVEL"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "Resume from:     $RESUME_CHECKPOINT"
fi
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; import numpy; import yaml; import gymnasium" || {
  echo "Error: Missing dependencies. Please install requirements:"
  echo "pip install -r requirements.txt"
  exit 1
}

echo "Dependencies OK"
echo ""

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

$CMD 