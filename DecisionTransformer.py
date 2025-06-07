import random
from dataclasses import dataclass
import numpy as np
import torch
import wandb
from datasets import load_dataset,load_from_disk
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments


@dataclass
class DecisionTransformerGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 20 #subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 1000 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["observations"]: # for all observations in the dataset
            states.extend(obs) #Add each observations to the states list
            traj_lens.append(len(obs)) #Add the lens of that obs

        self.n_traj = len(traj_lens)
        states = np.vstack(states)  #This vertically stack the arrays, so each state becomes a row

        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        print(f"Dataset obs mean: {self.state_mean}, std: {self.state_std}")

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        #Features is like a batch from dataset
        #so the len of features is a batch_size

        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:

            feature = self.dataset[int(ind)] # A feature is a row of s,a,r,d well whatever feature is on the dataset
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset, and turn it into a batch sequence
            s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))
            d.append(np.array(feature["terminals"][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1)) #However long the last sequence is the dataset

            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1] # trajectory_length = tlen

            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std # They do the normalisation after padding?
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }

class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1] # 1 is cause actions are at index 1 of the prediction head
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # reshapes action_preds to (total_items, action_dims) then applies mask
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

class ParamTrainableDT(DecisionTransformerModel):
    def __init__(self, config, loss_list): #This is a list containing (A,R,S)
        super().__init__(config)
        self.loss_list = loss_list

    def forward(self, **kwargs):

        total_loss = {}
        output = super().forward(**kwargs)

        state_preds = output[0]
        state_targets = kwargs["states"]

        action_preds = output[1]
        action_targets = kwargs["actions"]

        return_preds = output[2]
        return_targets = kwargs["returns_to_go"]

        attention_mask = kwargs["attention_mask"]

        state_dim = state_preds.shape[2]
        act_dim = action_preds.shape[2]
        ret_dim = return_preds.shape[2]

        action_pred = action_preds.reshape(-1,act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_targets.reshape(-1,act_dim)[attention_mask.reshape(-1) > 0]

        state_pred = state_preds.reshape(-1,state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_targets.reshape(-1,state_dim)[attention_mask.reshape(-1) > 0]

        return_pred = return_preds.reshape(-1,ret_dim)[attention_mask.reshape(-1) > 0]
        return_target = return_targets.reshape(-1,ret_dim)[attention_mask.reshape(-1) > 0]

        total_loss['A'] = torch.mean((action_pred - action_target) ** 2)
        total_loss['R'] = torch.mean((return_pred - return_target) ** 2)
        total_loss['S'] = torch.mean((state_pred - state_target) ** 2)

        loss = 0

        log_data = {l+"_LOSS" : total_loss[l] for l in self.loss_list}
        loss = sum([total_loss[l] for l in self.loss_list])
        wandb.log(log_data)
        
        # Log the total loss after the loop
        wandb.log({"total_loss": loss.item()})
        return {"loss": loss,
                "A_loss": total_loss['A'],
                "R_loss" : total_loss['R'],
                "S_loss" : total_loss['S'],
                "loss_details" : total_loss,
        }
        
    def original_forward(self, **kwargs):
        return super().forward(**kwargs)
