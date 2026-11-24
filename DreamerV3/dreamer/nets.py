"""Networks and distribution utilities for DreamerV3 (vector observations).

Follows Hafner et al. 2023 (arXiv:2301.04104): symlog predictions, twohot
critic/reward heads, categorical latents with unimix, KL balancing with
free bits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=2, act=nn.SiLU, norm=True, out_zero=False):
        super().__init__()
        seq = []
        d = in_dim
        for _ in range(layers):
            seq.append(nn.Linear(d, hidden, bias=not norm))
            if norm:
                seq.append(nn.LayerNorm(hidden))
            seq.append(act())
            d = hidden
        head = nn.Linear(d, out_dim)
        if out_zero:  # zero-init output layer (stabilizes heads)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        seq.append(head)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class TwoHot:
    """Twohot categorical regression head over symlog-spaced bins."""

    def __init__(self, num_bins=255, low=-20.0, high=20.0, device="cpu"):
        self.bins = torch.linspace(low, high, num_bins, device=device)  # symlog space
        self.num_bins = num_bins

    def to(self, device):
        self.bins = self.bins.to(device)
        return self

    def encode(self, y):
        """y: raw target -> twohot over bins in symlog space. Returns (..., num_bins)."""
        y = symlog(y).clamp(self.bins[0], self.bins[-1])
        idx = torch.searchsorted(self.bins, y)
        idx = idx.clamp(1, self.num_bins - 1)
        lo, hi = self.bins[idx - 1], self.bins[idx]
        w_hi = ((y - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        target = torch.zeros(*y.shape, self.num_bins, device=y.device)
        target.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        target.scatter_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def decode(self, logits):
        """logits: (..., num_bins) -> expected raw value."""
        probs = F.softmax(logits, -1)
        return symexp((probs * self.bins).sum(-1))

    def loss(self, logits, y):
        target = self.encode(y)
        return -(target * F.log_softmax(logits, -1)).sum(-1)


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical stochastic latents."""

    def __init__(self, embed_dim, action_dim, deter=256, stoch=16, classes=16,
                 hidden=256, unimix=0.01):
        super().__init__()
        self.deter, self.stoch, self.classes, self.unimix = deter, stoch, classes, unimix
        sd = stoch * classes
        self.img_in = MLP(sd + action_dim, hidden, hidden, 1)
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = MLP(deter, sd, hidden, 1)
        self.post_net = MLP(deter + embed_dim, sd, hidden, 1)

    def initial(self, batch, device):
        return (torch.zeros(batch, self.deter, device=device),
                torch.zeros(batch, self.stoch * self.classes, device=device))

    def _logits_to_dist(self, logits):
        logits = logits.view(*logits.shape[:-1], self.stoch, self.classes)
        probs = F.softmax(logits, -1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        return td.Independent(td.OneHotCategoricalStraightThrough(probs=probs), 1)

    def _sample(self, dist):
        s = dist.rsample()
        return s.reshape(*s.shape[:-2], self.stoch * self.classes)

    def img_step(self, deter, stoch, action):
        x = self.img_in(torch.cat([stoch, action], -1))
        deter = self.gru(x, deter)
        prior = self._logits_to_dist(self.prior_net(deter))
        return deter, prior

    def obs_step(self, deter, stoch, action, embed, is_first):
        # reset state at episode starts
        mask = (1.0 - is_first.float()).unsqueeze(-1)
        deter, stoch, action = deter * mask, stoch * mask, action * mask
        deter, prior = self.img_step(deter, stoch, action)
        post = self._logits_to_dist(self.post_net(torch.cat([deter, embed], -1)))
        return deter, post, prior

    def observe(self, embeds, actions, is_firsts, state=None):
        """embeds: (B,T,E), actions: (B,T,A) prev actions, is_firsts: (B,T).
        Returns deters (B,T,D), stochs (B,T,S), posts, priors (lists of dists)."""
        B, T = embeds.shape[:2]
        if state is None:
            deter, stoch = self.initial(B, embeds.device)
        else:
            deter, stoch = state
        deters, stochs, posts, priors = [], [], [], []
        for t in range(T):
            deter, post, prior = self.obs_step(deter, stoch, actions[:, t], embeds[:, t], is_firsts[:, t])
            stoch = self._sample(post)
            deters.append(deter); stochs.append(stoch); posts.append(post); priors.append(prior)
        return torch.stack(deters, 1), torch.stack(stochs, 1), posts, priors, (deter, stoch)

    @staticmethod
    def kl_loss(posts, priors, free=1.0, dyn_scale=0.5, rep_scale=0.1):
        dyn, rep = 0.0, 0.0
        for post, prior in zip(posts, priors):
            post_sg = td.Independent(td.OneHotCategoricalStraightThrough(probs=post.base_dist.probs.detach()), 1)
            prior_sg = td.Independent(td.OneHotCategoricalStraightThrough(probs=prior.base_dist.probs.detach()), 1)
            dyn = dyn + torch.clamp(td.kl_divergence(post_sg, prior), min=free)
            rep = rep + torch.clamp(td.kl_divergence(post, prior_sg), min=free)
        n = len(posts)
        return (dyn_scale * dyn + rep_scale * rep) / n  # (B,)


class Actor(nn.Module):
    def __init__(self, feat_dim, action_dim, discrete, hidden=256, layers=2,
                 unimix=0.01, min_std=0.1, max_std=1.0):
        super().__init__()
        self.discrete, self.unimix = discrete, unimix
        self.min_std, self.max_std = min_std, max_std
        out = action_dim if discrete else 2 * action_dim
        self.net = MLP(feat_dim, out, hidden, layers)

    def dist(self, feat):
        out = self.net(feat)
        if self.discrete:
            probs = F.softmax(out, -1)
            probs = (1 - self.unimix) * probs + self.unimix / probs.shape[-1]
            return td.OneHotCategoricalStraightThrough(probs=probs)
        mean, std = out.chunk(2, -1)
        mean = torch.tanh(mean)
        std = self.max_std * torch.sigmoid(std + 2.0) + self.min_std
        return td.Independent(td.Normal(mean, std), 1)

    def sample(self, feat, mode=False):
        d = self.dist(feat)
        if self.discrete:
            if mode:
                a = F.one_hot(d.probs.argmax(-1), d.probs.shape[-1]).float()
            else:
                a = d.rsample()
        else:
            a = d.base_dist.loc if mode else d.rsample()
            a = a.clamp(-1, 1)
        return a, d
