import torch


class CosineScheduler:
    """
    Implements a warmup scheduler that linearly increases LR from 0 to max,
    followed by a cosine annealing schedule.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance to schedule.
        warmup (int): Number of warmup steps.
        max_lr (float): Maximum learning rate to use.
        total_steps (int): Total number of steps including warmup.

    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup, max_lr: float, total_steps):
        self._optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, total_steps - warmup)
        self.warmup = warmup
        self._step = 0
        self.max_lr = max_lr
        self.total_steps = total_steps

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            for p in self._optimizer.param_groups:
                p['lr'] = self.warmup_func()
            self._optimizer.step()

        else:
            self._optimizer.step()
            self.scheduler.step()

    def warmup_func(self):
        return self.max_lr * self._step / self.warmup

    def test_linear_anneal(self):
        return self.max_lr-((self._step - self.warmup) * self.max_lr / (self.total_steps - self.warmup))

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_rate(self):
        for p in self._optimizer.param_groups:
            return p['lr']

    def get_state_dic(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):

        self.__dict__.update(state_dict)
