from copy import deepcopy

import torch
import torch._utils
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


class DDPBucketedParameters(torch.nn.Module):
    class Bucket:
        def __init__(self, ready: dict[str, bool], grad: torch.Tensor = None):
            self.ready = ready
            self.grad = grad

        @classmethod
        def from_names(cls, param_names: list[str]):
            return cls({name: False for name in param_names})

        def clear(self):
            for key in self.ready.keys():
                self.ready[key] = False
            self.grad = None

        def is_ready(self) -> bool:
            return all(self.ready.values())

        def __setitem__(self, key: str, value: bool):
            self.ready[key] = value

        def keys(self) -> list[str]:
            return list(self.ready.keys())

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = deepcopy(module)
        for param in self.module.parameters():
            # The broadcast happens in-place for the receiving tensors.
            dist.broadcast(param.data, src=0)

        self.communication_handles: dict[int, dist.Work] = {}
        self.buckets: list[DDPBucketedParameters.Bucket] = []
        self.params_map = {
            name: param for name, param in self.module.named_parameters() if param.requires_grad
        }

        # Gradient synchronization should be bucketed, with each bucket holding at most
        # bucket_size_mb of parameters
        bucket: list[str] = []
        current_size_bytes = 0
        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        params_list = list(self.module.named_parameters())
        params_list.reverse()
        for name, param in params_list:
            if param.requires_grad:
                cur_param_size_bytes = param.numel() * param.element_size()
                if current_size_bytes + cur_param_size_bytes > bucket_size_bytes:
                    self.buckets.append(DDPBucketedParameters.Bucket.from_names(bucket))
                    bucket = []
                    current_size_bytes = 0
                current_size_bytes += cur_param_size_bytes
                bucket.append(name)
        if bucket:
            self.buckets.append(DDPBucketedParameters.Bucket.from_names(bucket))

        for i, b in enumerate(self.buckets):
            for name in b.keys():
                param = self.params_map[name]
                param.register_post_accumulate_grad_hook(self._create_hook(i, name))

    def _create_hook(self, bucket_id: int, name: str):
        """
        Create a hook that will be called after the gradient is accumulated.
        """

        def hook(param):
            if param.grad is not None:
                bucket = self.buckets[bucket_id]
                bucket.ready[name] = True
                if bucket.is_ready():
                    flat_gradients = torch._utils._flatten_dense_tensors(
                        [self.params_map[name].grad.data for name in bucket.keys()]
                    )
                    bucket.grad = flat_gradients
                    handle = dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM, async_op=True)
                    self.communication_handles[bucket_id] = handle

        return hook

    def forward(self, *input, **kwargs):
        self.communication_handles.clear()
        for bucket in self.buckets:
            bucket.clear()
        return self.module(*input, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket_idx, handle in self.communication_handles.items():
            handle.wait()
            bucket = self.buckets[bucket_idx]
            unflattened_gradients = torch._utils._unflatten_dense_tensors(
                bucket.grad, [self.params_map[name].grad for name in bucket.keys()]
            )
            for name, grad in zip(bucket.keys(), unflattened_gradients):
                self.params_map[name].grad.data.copy_(grad / dist.get_world_size())

        self.communication_handles.clear()
        for bucket in self.buckets:
            bucket.clear()
