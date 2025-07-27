from copy import deepcopy

import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    """
    we can all-reduce parameter gradients as soon as they’re ready,
    reducing the overhead of data parallel training by overlapping computation of
    the backward pass with communication of gradients.

    - Backward hooks: To automatically call a function on a parameter
    after its gradient has been accumulated in the backward pass
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = deepcopy(module)
        for param in self.module.parameters():
            # The broadcast happens in-place for the receiving tensors.
            dist.broadcast(param.data, src=0)

        self.communication_handles: dict[str, dist.Work] = {}

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._create_hook(name))

    def _create_hook(self, name: str):
        """
        Create a hook that will be called after the gradient is accumulated.
        """

        def hook(param):
            if param.grad is not None:
                handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.communication_handles[name] = handle

        return hook

    def forward(self, *input, **kwargs):
        self.communication_handles.clear()
        return self.module(*input, **kwargs)

    def finish_gradient_synchronization(self):
        for param_name, handle in self.communication_handles.items():
            handle.wait()
            param = dict(self.module.named_parameters())[param_name]
            if param.grad is not None:
                param.grad.data /= dist.get_world_size()

        self.communication_handles.clear()
