import os
import gym
import wandb
import numpy as np
import wandb
import torch
from datasets import load_dataset,load_from_disk, concatenate_datasets
from transformers import Trainer, TrainingArguments
import DecisionTransformer 
import utils
from utils import get_action,run_episodes
from DecisionTransformer import DecisionTransformerGymDataCollator, TrainableDT, ParamTrainableDT
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments
import argparse
import gym
#import mujoco_py
#import d4rl

def download_datasets():
    for env_name in ["halfcheetah"]:
        for dataset_type in ["medium", "expert"]:
            name = f"{env_name}-{dataset_type}-v2"
            
            #delete old datasets
            if os.path.exists(f"Datasets/{name}"):
                os.system(f"rm -rf Datasets/{name}")

            #download new datasets
            dataset = load_dataset("edbeeching/decision_transformer_gym_replay", name)

            dataset.save_to_disk(f"Datasets/{name}")

#download_datasets()

#Concatenate datasets
def concatenate_datasets_():
    concatenated_dataset = None
    medium_dataset = load_from_disk("Datasets/halfcheetah-medium-v2")
    expert_dataset = load_from_disk("Datasets/halfcheetah-expert-v2")
    
    concatenated_dataset = concatenate_datasets([medium_dataset["train"], expert_dataset["train"]])

    concatenated_dataset.save_to_disk("Datasets/concatenated_halfcheetah")
    return concatenated_dataset
    
concatenate_datasets_()

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="DT_Haftcheeta_M_A")
parser.add_argument("--env", type=str, default="halfcheetah-medium-v2")
parser.add_argument("--outputs", nargs='+', type=str, required=True)
args = parser.parse_args()

directoy = "Datasets/" + args.env
train_dataset = load_from_disk(directoy)

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

collator = DecisionTransformerGymDataCollator(train_dataset["train"])

save_directory = "./outputs/" + args.name

loss_list = args.outputs

config = DecisionTransformerConfig(state_dim=collator.state_dim
                                  ,act_dim=collator.act_dim)

model = ParamTrainableDT(config,loss_list)

checkpoint_directory = save_directory + "_checkpoints"

if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

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
    train_dataset=train_dataset["train"],
    data_collator=collator, 
)

# if checkpoint_directory is not empty:
if os.path.isdir(checkpoint_directory) and (len(os.listdir(checkpoint_directory)) > 0) :
    print("resuming from checkpoint")
    trainer.train(resume_from_checkpoint=True)
else:
    print("training from scratch")
    trainer.train()

wandb.finish()

#model.save_pretrained(save_directory)
#
#env = gym.make("halfcheetah-medium-v2")
#max_ep_len = 1000
#device = "cpu"
#scale = 2655.0  # normalization for rewards/returns
#TARGET_RETURN = 10000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly
#
#model = TrainableDT.from_pretrained(save_directory)
#
#model = model.to("cpu")
#
#state_mean = collator.state_mean.astype(np.float32)
#state_std = collator.state_std.astype(np.float32)
#
#state_dim = env.observation_space.shape[0]
#act_dim = env.action_space.shape[0]
#
#state_mean = torch.from_numpy(state_mean).to(device=device)
#state_std = torch.from_numpy(state_std).to(device=device)
#
#EPISODES = 10
#
#return_list, length_list = run_episodes(EPISODES,TARGET_RETURN,device,env, model,act_dim,state_dim,state_mean,state_std,max_ep_len,scale)
#
#print(f"Average Return: {np.mean(return_list)}")
#print(f"Max Return: {np.max(return_list)}")
#print(f"Min Return: {np.min(return_list)}")
#print(f"Average Length: {np.mean(length_list)}")
#
#print("Hello World")