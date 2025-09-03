import torch 

import config
from env import game, get_mask, get_obs

def collect_rollouts(model, buffer):
    state = game.new_initial_state()

    while buffer.idx < len(buffer):
        obs_t = get_obs(state)
        mask_t = get_mask(state)

        action, value, logp = model.play(obs_t, mask_t)
        a = int(action.item())
        state.apply_action(a)

        buffer.store(
            obs=obs_t.flatten().cpu().numpy(),
            action=a,
            action_mask=mask_t.squeeze(0).to(torch.float32).cpu().numpy(),
            reward=0.0,
            logp=float(logp.item()),
            value=float(value.item()),
        )

        if state.is_terminal():
            if state.player_reward(0) == 1.0:
                winner = 1
            elif state.player_reward(1) == 1.0:
                winner = -1
            else:
                winner = 0
            buffer.calculate(winner=winner, last_value=0.0)

            state = game.new_initial_state()

    if not state.is_terminal():
        obs_t, mask_t = get_obs(state), get_mask(state)
        _, last_value, _ = model.play(obs_t, mask_t)
        buffer.calculate(winner=0, last_value=float(last_value.item()))