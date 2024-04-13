import torch
from torch import Tensor
from torchmetrics import Metric


from typing import Any


class QueryAccuracy(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0
    plot_upper_bound = 1

    compare_fns = {
        "exists": lambda preds, targets:  (preds > 0) == (targets > 0),
        "matches": lambda preds, targets:  preds == targets,
        "meets": lambda preds, targets:  preds >= targets
    }

    def __init__(self, criteria: str = "matches", **kwargs):
        super().__init__(**kwargs)
        assert criteria is not None
        assert criteria in self.compare_fns
        self.compare_fn = self.compare_fns[criteria]
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")
        
        self.correct += torch.sum(self.compare_fn(preds, targets))
        self.total += targets.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    def plot(self, val: Any = None, ax: Any = None) -> Any:
        return self._plot(val, ax)