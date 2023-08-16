import matplotlib.pyplot as plt
import torch.nn

from optimizer.cosine_shceduler import CosineScheduler
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def inverse_square_root_anneal(model_size, warmup, step=None):
    return (model_size ** (-0.5) *
            min(step ** (-0.5), step * warmup ** (-1.5)))


class TestModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(1, 2)

    def forward(x, self):
        return self.fc(x)


model = TestModel().cuda()
op = torch.optim.Adam(model.parameters())
scheduler = CosineScheduler(op, 2500, 1e-3, 20000)

steps = range(1, 20000)
rates1 = []
rates2 = []
rates3 = []
for i in steps:
    scheduler.zero_grad()
    scheduler.step()
    if i > 2500:
        rates1.append(scheduler.test_linear_anneal())
    else:
        rates1.append(scheduler.get_rate())
    rates2.append(scheduler.get_rate())
    rates3.append(inverse_square_root_anneal(128, 2500, i))
plt.plot(steps, rates1, label='Linear Annealing')
plt.plot(steps, rates2, label='Cosine Annealing')
plt.plot(steps, rates3, label='ISR Annealing (dim=128)')

plt.xlabel('Step')
plt.ylabel('Rate')
plt.title('Rates following Steps')
plt.legend()
plt.grid(True)
plt.show()
