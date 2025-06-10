import os
import gym
import wandb
import numpy as np
import wandb
import torch
from datasets import load_dataset,load_from_disk, concatenate_datasets, DatasetDict, Dataset
from transformers import Trainer, TrainingArguments
import DecisionTransformer 
import utils
from utils import get_action,run_episodes
from DecisionTransformer import DecisionTransformerGymDataCollator, TrainableDT, ParamTrainableDT
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
import argparse
import gym
import datasets
import h5py
#import mujoco_py
import d4rl


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="DT_Haftcheetah_A")
parser.add_argument("--env", type=str, default="halfcheetah_medium-v2")
parser.add_argument("--outputs", nargs='+', type=str, required=True)
args = parser.parse_args()

file = "Datasets/"+ args.env +".hdf5"
file = h5py.File(file, 'r')

#convert to numpy array 
actions = file['actions'][:]
observations = file['observations'][:]
rewards = file['rewards'][:]
terminals = file['terminals'][:]
#create hugging face dataset
train_dataset = Dataset.from_dict({
    'actions': actions,
    'observations': observations,
    'rewards': rewards,
    'terminals': terminals,
})

def convert_to_trajactories(dataset):
    actions = dataset['actions']
    observations = dataset['observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    
    actions_set = []
    observations_set = []
    rewards_set = []
    terminals_set = []
    action_traj = []
    observation_traj = []
    reward_traj = []
    terminal_traj = []

    trajectory_count = 0
    
    for i in range(len(terminals)):
        action_traj.append(actions[i])
        observation_traj.append(observations[i])
        reward_traj.append(rewards[i])
        terminal_traj.append(terminals[i])
        if len(terminal_traj) == 1000:
            actions_set.append(action_traj)
            observations_set.append(observation_traj)
            rewards_set.append(reward_traj)
            terminals_set.append(terminal_traj)
            action_traj = []
            observation_traj = []
            reward_traj = []
            terminal_traj = []
            trajectory_count += 1

    return_dataset = Dataset.from_dict({
        'actions': actions_set,
        'observations': observations_set,
        'rewards': rewards_set,
        'terminals': terminals_set,
    })
    return return_dataset

train_dataset = convert_to_trajactories(train_dataset)

os.environ["WANDB_MODE"] = "offline"

wandb.init(
    project="DT",
    entity="luthando_nxumalo-university-of-kwazulu-natal",
    name=args.name,
    mode ="offline",
    config={
        "env":args.env,
        "outputs":args.outputs
    }
)

collator = DecisionTransformerGymDataCollator(train_dataset)

save_directory = "./outputs/" + args.name

loss_list = args.outputs

config = DecisionTransformerConfig(state_dim=collator.state_dim
                                  ,act_dim=collator.act_dim)

model = ParamTrainableDT(config,loss_list)

checkpoint_directory = save_directory + "_checkpoints"

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

## TODO : Add early stopping 
training_args = TrainingArguments(output_dir=checkpoint_directory,
    run_name= args.name,
    remove_unused_columns=False,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=10,
    max_steps=5,
    logging_steps=1000,
    report_to="wandb",
    logging_dir="wandb/"+args.name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator, 
)

# if checkpoint_directory is not empty:
if os.path.isdir(checkpoint_directory) and (len(os.listdir(checkpoint_directory)) > 0) :
    print("resuming from checkpoint")
    trainer.train(resume_from_checkpoint=True)
else:
    print("training from scratch")
    trainer.train()

#load the dataset





wandb.finish()