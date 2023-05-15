#!/bin/bash -l

# Wandb API key.
WANDB_KEY=$(<wandb_key)

BASE_PATH="/projectnb/llamagrp/feyzanb/feedback/alphabetize_output"
PROJECT_NAME="rl4f_alphabetize_ppo"
EXPERIMENT_NAME="t5large"
mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME
WANDB_API_KEY=$WANDB_KEY python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/alphabetize/t5large_ppo_on_supervised.yaml \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name feyzaakyurek \
--log_to_wandb # > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/$EXPERIMENT_NAME.log 2>&1
