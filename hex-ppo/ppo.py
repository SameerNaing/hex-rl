import torch
from config import CLIP_EPS, TRAIN_STEPS

def calculate_loss(data, model, c1=0.5, c2=0.01, clip_eps=CLIP_EPS):
    obs, act, act_mask, adv, logp_old, ret = (
        data['obs'], data['act'], data['act_mask'], data['adv'], data['logp'], data['ret']
    )
    
    dist, value = model(obs, act_mask.bool())
    logp = dist.log_prob(act)
    ratio = torch.exp(logp - logp_old)

    clip_obj = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    clip_loss = torch.min(ratio * adv, clip_obj).mean()
    value_loss = ((value - ret) ** 2).mean()
    entropy = dist.entropy().mean()

    loss = -clip_loss + c1 * value_loss - c2 * entropy
    approx_kl = (logp_old - logp).mean().item()
    clipfrac = (torch.abs(ratio - 1) > clip_eps).float().mean().item()
    
    return loss, dict(kl=approx_kl, ent=entropy.item(), clipped_fraction=clipfrac)

def update(optimizer, model, data, train_steps=TRAIN_STEPS):
    last_loss, last_info = None, None
    for i in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss, info = calculate_loss(data, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        last_loss, last_info = loss, info
        
        if i >= 50 and info['kl'] > 1.5:
            break
        
    return float(last_loss.item()), last_info
