import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def pprint_size(data_size_bytes: int) -> str:
    """Pretty print the size in bytes."""
    units = [(1024**3, "GB"), (1024**2, "MB"), (1024, "KB"), (1, "bytes")]

    for size, unit in units:
        if data_size_bytes >= size:
            if unit == "bytes":
                return f"{data_size_bytes} {unit}"
            return f"{data_size_bytes / size:.2f} {unit}"


def distributed_demo(rank, world_size, backend="gloo"):
    setup(rank, world_size, backend)
    # create a tensor float32 with random values
    data = torch.randn(1024 // 4, 1024, dtype=torch.float32)
    data_size_bytes = data.numel() * data.element_size()

    print(f"rank: {rank} data size: {pprint_size(data_size_bytes)}")
    if backend == "nccl":
        data = data.to(f"cuda:{rank}")
    print(f"rank: {rank} data device: {data.device}")
    print(f"rank {rank} data (before all-reduce): {data}")

    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    end_time = timeit.default_timer()
    duration = end_time - start_time
    print(f"rank {rank} data (after all-reduce): {data}")
    print(f"rank {rank} all-reduce time: {duration:.6f} seconds")
    # aggregate duration across all ranks using all_gather_object
    gathered_durations = [0] * world_size
    dist.all_gather_object(gathered_durations, duration)
    # calculate the average duration
    avg_duration = sum(gathered_durations) / world_size
    print(
        f"rank {rank} average all-reduce time across all ranks: {avg_duration:.6f} seconds"
        f" ({pprint_size(data_size_bytes)})"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(fn=distributed_demo, args=(world_size, "gloo"), nprocs=world_size, join=True)
