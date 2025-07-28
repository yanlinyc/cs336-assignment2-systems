from collections.abc import Callable

import torch.distributed as dist
from pyparsing import Any
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    A sharded optimizer that synchronizes gradients across multiple processes.
    """

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        super().__init__(params, {})

        params_by_rank = {rank: [] for rank in range(self.world_size)}
        for _, param_group in enumerate(self.param_groups):
            params = param_group["params"]
            for param_idx, param in enumerate(params):
                owner_rank = param_idx % self.world_size
                params_by_rank[owner_rank].append(param)
        self.params_by_rank = params_by_rank

    def add_param_group(self, param_group):
        """
        Adds a parameter group to the sharded optimizer and shards the parameters.
        """
        super().add_param_group(param_group)

        # Shard the parameters in the new group
        params = param_group["params"]
        sharded_params = [p for i, p in enumerate(params) if i % self.world_size == self.rank]

        # Create a new parameter group for the sharded parameters
        sharded_param_group = {k: v for k, v in param_group.items() if k != "params"}
        sharded_param_group["params"] = sharded_params

        # If the optimizer is not yet created, create it now
        if not hasattr(self, "optimizer"):
            self.optimizer = self.optimizer_cls([sharded_param_group], **self.optimizer_kwargs)
        else:
            self.optimizer.add_param_group(sharded_param_group)

    def step(self, closure: Callable | None = None, **kwargs):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                           and returns the loss.
        """
        # Step 1: Call the wrapped optimizer's step() method
        loss = self.optimizer.step(closure, **kwargs)

        # Broadcast all parameters from each rank
        for owner_rank, rank_params in self.params_by_rank.items():
            for param in rank_params:
                dist.broadcast(param.data, src=owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
