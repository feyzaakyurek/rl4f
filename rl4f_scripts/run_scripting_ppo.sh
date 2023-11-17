#!/bin/bash -l

# Make sure to edit:
# - model_name in config yml file in config_path to point to the ../model folder.
#   or if using checkpoint to t5large_openai_summ_QMQ3V
# - data file paths in rl4lms/data_pools/custom_text_generation_pools.py
# - prompt_path (x2) in config file to point to the ../prompts_edit_numeric.txt
# - cache_path (x2) for storing generations in config file.

BASE_PATH="./interscript_output"
PROJECT_NAME="interscript_ppo_scale"
EXPERIMENT_NAME="t5large_lr5e7_pet"

mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME

WANDB_API_KEY=INSERT_KEY_HERE python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/scripting/t5large_ppo_on_supervised_scale.yml \
--base_path_to_store_results $BASE_PATH \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name feyzaakyurek \
--log_to_wandb > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/log.out 2>&1

