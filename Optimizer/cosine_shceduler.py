class CosineScheduler:
    """This scheduler first make the learning rate grow linearly from zero to the maximum. Then it will gradually reduce
    by cosine anneal which must specify total steps."""
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup, max_lr: float, total_steps):
        self._optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, total_steps - warmup)
        self.warmup = warmup
        self._step = 0
        self.max_lr = max_lr

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

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_rate(self):
        for p in self._optimizer.param_groups:
            return p['lr']

    def get_state_dic(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):

        self.__dict__.update(state_dict)
