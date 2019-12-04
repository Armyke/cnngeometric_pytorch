import math

from torch.optim.lr_scheduler import _LRScheduler


class TruncateCosineScheduler(_LRScheduler):

    def __init__(self, optimizer,
                 n_epochs: int, n_cycles: int,
                 annealing: bool = True,
                 last_epoch=-1):
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.annealing = annealing
        self.last_epoch = last_epoch
        super(TruncateCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        param = 1
        epochs_per_cycle = math.floor(self.n_epochs / self.n_cycles)

        if self.annealing:
            param = 1 + self.last_epoch / self.n_epochs
            epochs_per_cycle *= 1 + param

        cos_inner = math.pi * (self.last_epoch % epochs_per_cycle) / epochs_per_cycle

        return [base_lr / (2 * param) * (math.cos(cos_inner) + 1)
                for base_lr in self.base_lrs]


def cosine_lr_func_gen(n_epochs: int, n_cycles: int, lrate_max: float,
                       annealing: bool = True):
    """Generates a learning rate function of truncated cosine type,
       if annealing is True (default) as epochs progress
       amplitude decreases and cycle length increases, otherwise cycle remains the same"""

    def cosine_learning_rate(epoch):
        """Function generated with the parameters retrieved from cosine_lr_func_gen"""

        param = 1
        epochs_per_cycle = math.floor(n_epochs / n_cycles)

        if annealing:
            param = 1 + epoch / n_epochs
            epochs_per_cycle *= 1 + param

        cos_inner = math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle

        new_lr = lrate_max/(2 * param) * (math.cos(cos_inner) + 1)

        return new_lr

    return cosine_learning_rate


if __name__ == '__main__':

    from argparse import ArgumentParser
    import matplotlib.pyplot as plt

    PARSER = ArgumentParser()
    PARSER.add_argument('--lr_max', default=0.001, type=float)
    PARSER.add_argument('--cycles', default=15, type=int)
    PARSER.add_argument('--steps', default=1000, type=int)
    ARGS = PARSER.parse_args()

    LR_FUNC1 = cosine_lr_func_gen(10, 5, 0.5)
    LR_FUNC2 = cosine_lr_func_gen(10, 5, 0.5, annealing=False)

    LR_EVOLUTION1 = [LR_FUNC1(i) for i in range(100)]
    LR_EVOLUTION2 = [LR_FUNC2(i) for i in range(100)]

    plt.plot([i for i in range(100)], LR_EVOLUTION1)
    plt.plot([i for i in range(100)], LR_EVOLUTION2)
    plt.show()
