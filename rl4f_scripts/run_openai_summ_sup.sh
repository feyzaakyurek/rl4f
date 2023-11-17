#!/bin/bash -l

# Output path - where checkpoints and predictions are saved.
BASE_PATH="openai_summ_output"

# Project name and experiment name - used for wandb logging and output folder.
PROJECT_NAME="openai_summ_sup"
EXPERIMENT_NAME="t5large_lr1e5"
EXPERIMENT_FOLDER=$BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME

mkdir -p $EXPERIMENT_FOLDER

WANDB_API_KEY=INSERT_WANDB_KEY_HERE python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_supervised.yml \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name feyzaakyurek \
--log_to_wandb > $EXPERIMENT_FOLDER/log.out 2>&1