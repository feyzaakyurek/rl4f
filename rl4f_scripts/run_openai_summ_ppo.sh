#!/bin/bash -l

# Make sure to edit:
# - model_name in config yml file in config_path to point to the ../model folder.
#   or if using checkpoint to t5large_openai_summ_QMQ3V
# - data file paths in rl4lms/data_pools/custom_text_generation_pools.py
# - prompt_path (x2) in config file to point to the ../prompts_edit_instructional_novo.txt
# - cache_path (x2) for storing generations in config file.

# Output path.
BASE_PATH="./openai_summ_output"

# Project name and experiment name - used for wandb logging and output folder.
PROJECT_NAME="openai_summ"
EXPERIMENT_NAME="t5large_rl4lm_lr10e6_rougeC_supervised_gen_args_me_48g"

mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME


WANDB_API_KEY=INSERT_WANDB_KEY_HERE python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_ppo_on_supervised.yml \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name feyzaakyurek \
--log_to_wandb > $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/log.out 2>&1
