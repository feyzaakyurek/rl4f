# RL4F: Generating Natural Language Feedback with Reinforcement Learning

Code repository for the ACL 2023 paper "RL4F: Generating Natural Language Feedback with Reinforcement Learning". Afra Feyza Akyurek, Ekin Akyurek, Aman Madaan, Ashwin Kalyan, Peter Clark, Derry Wijaya, Niket Tandon. 

This codebase is primarily based on the [RL4LMs](https://github.com/allenai/RL4LMs) repository. We provide custom data classes and reward functions to implement RL4F.

# Installation

```bash
git clone https://github.com/feyzaakyurek/rl4f.git
cd rl4f
pip install -e .
```

# Introduction to the Codebase

`rl4lms/data_pools/custom_text_generation_pools.py`: This file contains custom dataset loading classes. Make sure to specify the correct data paths in respective classes.

`scripts/training/task_configs`: Yaml files containing configs are stored under this path. This is where we specify the training and evaluation arguments, model and output paths. 

`rl4f_scripts/run_alphabetize_sup.sh`: A sample sh script for supervised critique generation.

`openai_key`: If running RL4F with one of OpenAI models, you need to place your API key in a file and specify the path in config. Give a path to this key in your yaml file. *Note that RL4F runs using openai API might incur significant charges.*

`wandb_key`: We track our runs using wandb, specify your API key here which is used in the sh script.


# Running Experiments

## Datasets
Download the pretrained checkpoints and data from this [link](https://drive.google.com/drive/folders/1Rl5j7r8RqvOhQUQPRhK8AEoD5-bjAuDI?usp=sharing). For topic-based summarization dataset check out [Saunders et al. (2022)](https://arxiv.org/abs/2206.05802).

All scripts can be found under `rl4f_scripts`. For example, check out the `rl4f_scripts/run_alphabetize_sup.sh` script for warm-starting a pretrained T5-large for supervised critique generation for alphabetization. Alternatively, you can load the released checkpoint from the above drive link. For PPO training, specify the checkpoint at `scripts/training/task_configs/alphabetize/t5large_ppo_on_supervised.yaml` and run `rl4f_scripts/run_alphabetize_ppo.sh`.


# FAQ
If you are receiving an error about your torch installation not supporting sm_86, try uninstalling torch and reinstalling with conda using the cudatoolkit that matches your environment. E.g.

```
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
```

# Bibtex
