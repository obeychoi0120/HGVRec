#!/usr/bin/env bash
# End-to-end pipeline template. Copy, edit the variables, and comment out steps you do not need:
#   cp run_script.example.sh run_script.sh && bash run_script.sh
# run_script*.sh is git-ignored, so your copy stays local.
# Every step skips finished outputs; add --force to a step to rebuild it.
set -euo pipefail
cd "$(dirname "$0")"

# ---- Settings -------------------------------------------------------------
RUN_ID=exp001                    # output: artifacts/runs/$RUN_ID/$DATASET/
DATASET=microlens_100k           # microlens_100k | m3l_10m
MODEL=qwen3vl_2b                 # qwen3vl_2b | qwen3vl_4b | qwen3vl_8b | gemini
WORKERS_PER_GPU=4                # concurrent SASRec trainings per GPU (1 on CPU)
GRAPH_PROMPT=prompts/scene_graph_v8.md
DESC_PROMPT=prompts/scene_description_dwys.md

# export CUDA_VISIBLE_DEVICES=0,1                    # GPUs to use (default: all visible)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # works without `pip install -e .`
export VLLM_PLUGINS=""
export VLLM_LOGGING_LEVEL=ERROR

COMMON=(--run-id "$RUN_ID" --dataset "$DATASET")

# ---- Input: cohort and keyframes -------------------------------------------
python -m preparation prepare-cohort "${COMMON[@]}"
python -m preparation prepare-input-data "${COMMON[@]}"

# ---- Step 1: scene extraction (sources used in rec_experiments_config.yaml) -----
# Main experiments: on-device Qwen3-VL ($MODEL = qwen3vl_*).
python -m extraction extract-graph-scenes "${COMMON[@]}" --model "$MODEL" --schema "$GRAPH_PROMPT" --repetition-penalty 1.05 1.10 1.15 --stop-repeated-lines
python -m extraction extract-description-scenes "${COMMON[@]}" --model "$MODEL" --schema "$DESC_PROMPT"
# Optional Qwen-only generation options:
#   --presence-penalty 1.5 | --repetition-penalty 1.05 1.1 | --stop-repeated-lines (graph only)

# Additional experiments: Gemini on Vertex AI (requirements-extra.txt, gcloud ADC login).
# python -m extraction extract-graph-scenes "${COMMON[@]}" --model gemini --schema "$GRAPH_PROMPT"
# python -m extraction extract-description-scenes "${COMMON[@]}" --model gemini --schema "$DESC_PROMPT"

# ---- Step 2-3: frozen BGE features and module inputs (ASHA segments) --------
python -m validation embed-representations "${COMMON[@]}"
python -m validation prepare-module "${COMMON[@]}"

# ---- Step 4: recommendation training and diagnosis -------------------------
python -m validation run-recommendation "${COMMON[@]}" --workers-per-gpu "$WORKERS_PER_GPU"
python -m validation run-diagnosis "${COMMON[@]}"
# Compare with another Run that has the same cohort and splits:
# python -m validation run-diagnosis "${COMMON[@]}" --compare-run-id <REFERENCE_RUN_ID>
