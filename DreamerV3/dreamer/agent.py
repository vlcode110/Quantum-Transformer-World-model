"""DreamerV3 agent: world model + actor-critic trained in imagination."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import MLP, RSSM, Actor, TwoHot, symlog


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg, device):
        super().__init__()
        h = cfg["hidden"]
        self.encoder = MLP(obs_dim, cfg["embed"], h, cfg["layers"])
        self.rssm = RSSM(cfg["embed"], action_dim, cfg["deter"], cfg["stoch"],
                         cfg["classes"], h)
        feat = cfg["deter"] + cfg["stoch"] * cfg["classes"]
        self.feat_dim = feat
        self.decoder = MLP(feat, obs_dim, h, cfg["layers"])
        self.reward_head = MLP(feat, cfg["bins"], h, cfg["layers"], out_zero=True)
        self.cont_head = MLP(feat, 1, h, cfg["layers"])
        self.twohot = TwoHot(cfg["bins"], device=device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg["model_lr"], eps=1e-8)
        self.clip = cfg["grad_clip"]

    @staticmethod
    def feat(deter, stoch):
        return torch.cat([deter, stoch], -1)

    def train_step(self, batch):
        embed = self.encoder(symlog(batch["obs"]))
        deters, stochs, posts, priors, last_state = self.rssm.observe(
            embed, batch["action"], batch["is_first"])
        feat = self.feat(deters, stochs)

        recon = self.decoder(feat)
        recon_loss = ((recon - symlog(batch["obs"])) ** 2).sum(-1).mean()
        reward_loss = self.twohot.loss(self.reward_head(feat), batch["reward"]).mean()
        cont_loss = F.binary_cross_entropy_with_logits(
            self.cont_head(feat).squeeze(-1), batch["cont"]).mean()
        kl = self.rssm.kl_loss(posts, priors).mean()

        loss = recon_loss + reward_loss + cont_loss + kl
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        self.opt.step()

        metrics = {"wm/recon": recon_loss.item(), "wm/reward": reward_loss.item(),
                   "wm/cont": cont_loss.item(), "wm/kl": kl.item()}
        return (deters.detach(), stochs.detach()), metrics


class ActorCritic(nn.Module):
    def __init__(self, feat_dim, action_dim, discrete, cfg, device):
        super().__init__()
        h = cfg["hidden"]
        self.actor = Actor(feat_dim, action_dim, discrete, h, cfg["layers"])
        self.critic = MLP(feat_dim, cfg["bins"], h, cfg["layers"], out_zero=True)
        self.slow_critic = MLP(feat_dim, cfg["bins"], h, cfg["layers"], out_zero=True)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.twohot = TwoHot(cfg["bins"], device=device)
        self.discrete = discrete
        self.horizon = cfg["horizon"]
        self.gamma = cfg["gamma"]
        self.lam = cfg["lam"]
        self.ent_scale = cfg["ent_scale"]
        self.slow_tau = cfg["slow_tau"]
        self.clip = cfg["grad_clip"]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"], eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"], eps=1e-8)
        # return-normalization EMA of the 5th-95th percentile range
        self.register_buffer("ret_scale", torch.ones(()))

    def imagine(self, wm, start_deter, start_stoch):
        """Roll out the actor in latent space. start_*: (N, D)/(N, S)."""
        deter, stoch = start_deter, start_stoch
        feats, actions, ents = [], [], []
        for _ in range(self.horizon):
            feat = wm.feat(deter, stoch)
            action, dist = self.actor.sample(feat)
            ent = dist.entropy() if self.discrete else dist.base_dist.entropy().sum(-1)
            deter, prior = wm.rssm.img_step(deter, stoch, action)
            stoch = wm.rssm._sample(prior)
            feats.append(feat); actions.append(action); ents.append(ent)
        feats.append(wm.feat(deter, stoch))
        return torch.stack(feats), torch.stack(actions), torch.stack(ents)  # (H+1,N,F),(H,N,A),(H,N)

    def lambda_returns(self, rewards, conts, values):
        """rewards/conts/values: (H,N),(H,N),(H+1,N) -> returns (H,N)."""
        rets = []
        last = values[-1]
        for t in reversed(range(rewards.shape[0])):
            last = rewards[t] + self.gamma * conts[t] * (
                (1 - self.lam) * values[t + 1] + self.lam * last)
            rets.append(last)
        return torch.stack(list(reversed(rets)))

    def train_step(self, wm, start_deter, start_stoch):
        # flatten (B,T) starts -> N
        N = start_deter.shape[0] * start_deter.shape[1]
        deter = start_deter.reshape(N, -1)
        stoch = start_stoch.reshape(N, -1)

        feats, actions, ents = self.imagine(wm, deter, stoch)
        # Keep the graph: for continuous actions the actor gradient flows
        # through rewards/values via imagined dynamics (DreamerV3 backprop mode).
        rewards = wm.twohot.decode(wm.reward_head(feats[:-1]))
        conts = torch.sigmoid(wm.cont_head(feats)).squeeze(-1)
        values = self.twohot.decode(self.critic(feats))
        rets = self.lambda_returns(rewards, conts[:-1],
                                   values.detach() if self.discrete else values)
        # discount weights: don't train past predicted episode end
        with torch.no_grad():
            w = torch.cumprod(torch.cat([torch.ones_like(conts[:1]),
                                         self.gamma * conts[:-2]], 0), 0)

        # --- return normalization (percentile EMA) ---
        with torch.no_grad():
            lo = torch.quantile(rets, 0.05)
            hi = torch.quantile(rets, 0.95)
            scale = (hi - lo).clamp(min=1.0)
            self.ret_scale.mul_(0.99).add_(0.01 * scale)

        # --- actor ---
        adv = (rets - values[:-1]).detach() / self.ret_scale
        dists = self.actor.dist(feats[:-1].detach())
        if self.discrete:
            logp = dists.log_prob(actions.detach())
            actor_loss = -(w * (logp * adv + self.ent_scale * ents)).mean()
        else:
            # dynamics backprop: returns are differentiable through imagination
            norm_rets = rets / self.ret_scale
            actor_loss = -(w * (norm_rets + self.ent_scale * ents)).mean()
        # Backward deposits gradients in wm/critic params too (continuous mode);
        # those are cleared by their own zero_grad() before their next steps.
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.actor_opt.step()

        # --- critic (twohot regression toward lambda-returns + slow reg) ---
        feats_d = feats[:-1].detach()
        critic_logits = self.critic(feats_d)
        critic_loss = (w * self.twohot.loss(critic_logits, rets.detach())).mean()
        with torch.no_grad():
            slow_target = self.twohot.decode(self.slow_critic(feats_d))
        critic_loss = critic_loss + (w * self.twohot.loss(critic_logits, slow_target)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        self.critic_opt.step()

        # slow critic EMA update
        with torch.no_grad():
            for p, sp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                sp.mul_(1 - self.slow_tau).add_(self.slow_tau * p)

        return {"ac/actor": actor_loss.item(), "ac/critic": critic_loss.item(),
                "ac/return": rets.mean().item(), "ac/value": values.mean().item(),
                "ac/entropy": ents.mean().item(), "ac/ret_scale": self.ret_scale.item()}


class Dreamer(nn.Module):
    def __init__(self, obs_dim, action_dim, discrete, cfg, device):
        super().__init__()
        self.wm = WorldModel(obs_dim, action_dim, cfg, device)
        self.ac = ActorCritic(self.wm.feat_dim, action_dim, discrete, cfg, device)
        self.device = device
        self.to(device)

    def train_step(self, batch):
        (deters, stochs), m1 = self.wm.train_step(batch)
        m2 = self.ac.train_step(self.wm, deters, stochs)
        return {**m1, **m2}

    @torch.no_grad()
    def policy(self, obs, state, mode=False):
        """Single-env acting. state = (deter, stoch, prev_action) or None."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if state is None:
            deter, stoch = self.wm.rssm.initial(1, self.device)
            prev_action = torch.zeros(1, self.ac.actor.net.net[-1].out_features
                                      if self.ac.discrete else
                                      self.ac.actor.net.net[-1].out_features // 2,
                                      device=self.device)
            is_first = torch.ones(1, device=self.device)
        else:
            deter, stoch, prev_action = state
            is_first = torch.zeros(1, device=self.device)
        embed = self.wm.encoder(symlog(obs))
        deter, post, _ = self.wm.rssm.obs_step(deter, stoch, prev_action, embed, is_first)
        stoch = self.wm.rssm._sample(post)
        feat = self.wm.feat(deter, stoch)
        action, _ = self.ac.actor.sample(feat, mode=mode)
        return action.squeeze(0).cpu().numpy(), (deter, stoch, action)
