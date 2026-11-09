"""Non-monotonically Triggered AvSGD (Merity et al., 2018, Algorithm 1)."""
from contextlib import contextmanager

import torch


class NTAvSGD:
    """Wraps an SGD optimizer and, until the trigger, it is exactly SGD. After every validation check (one per
    epoch, i.e. logging interval L = 1 epoch) the dev perplexity v is compared with the logs: averaging is triggered
    the first time that

        t > n  and  v > min(logs[:t - n])

    i.e. v is worse than the best perplexity obtained more than n checks ago (n = non-monotone interval), so a single
    bad epoch is not enough. From the trigger point T on, the weights w_T, ..., w_k produced by every SGD step are
    averaged with a running mean, and the AVERAGED weights are the ones used for validation, model selection and
    test.
    """

    def __init__(self, model, optimizer, non_mono=5):
        self.model = model
        self.optimizer = optimizer
        self.non_mono = non_mono
        self.logs = []             # dev perplexities collected before the trigger
        self.trigger_epoch = None
        self.avg_params = None     # running average of the weights, created at the trigger
        self.n_averaged = 0

    @property
    def triggered(self):
        return self.avg_params is not None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()  # plain SGD step
        if self.triggered:
            self.n_averaged += 1
            with torch.no_grad():
                for avg, param in zip(self.avg_params, self.model.parameters(), strict=True):
                    avg.add_(param - avg, alpha=1.0 / self.n_averaged)  # avg_k = avg_k-1 + (w_k - avg_k-1) / k

    def check(self, ppl_dev, epoch):
        """Validation check, called once per epoch before the trigger. Returns True if it triggers now."""
        t = len(self.logs)
        if not self.triggered and t > self.non_mono and ppl_dev > min(self.logs[:t - self.non_mono]):
            # The average starts from the current weights w_T
            self.avg_params = [param.detach().clone() for param in self.model.parameters()]
            self.n_averaged = 1
            self.trigger_epoch = epoch
        self.logs.append(ppl_dev)
        return self.trigger_epoch == epoch

    @contextmanager
    def averaged_weights(self):
        """Temporarily copy the averaged weights into the model (e.g. to evaluate or save them)."""
        if not self.triggered:
            yield
            return
        backup = [param.detach().clone() for param in self.model.parameters()]
        with torch.no_grad():
            for param, avg in zip(self.model.parameters(), self.avg_params, strict=True):
                param.copy_(avg)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), backup, strict=True):
                    param.copy_(saved)
