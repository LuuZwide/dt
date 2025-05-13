import torch
import numpy as np

# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim) # Reshaping states to (1, num_timesteps, state_shpe)
    actions = actions.reshape(1, -1, model.config.act_dim) # Reshaping actions to (1, num_timesteps, action_shape)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1) #[[t1,t2, t3......]] -> 1 row x amount of columns

    states = states[:, -model.config.max_length :] # take all rows; last max length timesteps; all columns
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1] # Paddiing equal max_leng - how many timesteps now example padding = 20 - timesteps -> n
    # pad all tokens to sequence length

    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])]) #[0,0,...., 1,1,1] 0 where there is padding and 1 where timesteps
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)#[[0,0, ....1,1,1]
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float() # Joins them on the timestep dimension
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    #Example shape
    #action shape : torch.Size([1, 20, 6])
    #states shape : torch.Size([1, 20, 17])
    #returns_to_go shape : torch.Size([1, 20, 1])
    #timesteps shape : torch.Size([1, 20])

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

def run_episodes(num_episodes,TARGET_RETURN,device, env, model,act_dim,state_dim,state_mean,state_std,max_ep_len,scale):

    ran_episodes_returns = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    for i in range(num_episodes):

      episode_return, episode_length = 0, 0
      state = env.reset()
      target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1) #[[1000]] a single value
      states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32) # The reset state (1,state_stape,) -> [[X,Y,Z....]]
      actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32) #Initial actions are (action_dim) [0,0,0,..]
      rewards = torch.zeros(0, device=device, dtype=torch.float32) #Initial rewards is a 0
      timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1) #[[0]]

      for t in range(max_ep_len):

          actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0) # [actions, actions, [[0,0,0..]] ] #Padding a zeros to actions
          rewards = torch.cat([rewards, torch.zeros(1, device=device)]) # [rewards,rewards, [0]] #Padding [0] to the rewards

          action = get_action(
              model,
              (states - state_mean) / state_std,
              actions,
              rewards,
              target_return,
              timesteps,
          )
          actions[-1] = action # action is set to last value in the actions array i.e the padded zeros above
          action = action.detach().cpu().numpy()

          state, reward, done, _ = env.step(action)

          cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim) # Reshape state to [[X,Y,Z...]]
          states = torch.cat([states, cur_state], dim=0) # Add the recieved state from the environment into the states array [states, states, ... curr_states]
          rewards[-1] = reward # set the last reward to the recieved reward

          pred_return = target_return[0, -1] - (reward / scale) # [0, -1] -> row 0 last column; basically the last target_return - (recieved reward)
          target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1) # Add to the target returns [ [tr_0], [tr_1], [tr_2], ......]
          timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1) # increase timesteps [[t0], [t0 + 1]]

          episode_return += reward
          episode_length += 1

          if done:
              ran_episodes_returns[i] = episode_return
              episode_lengths[i] = episode_length
              break

    return ran_episodes_returns, episode_lengths