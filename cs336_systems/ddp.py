import os
import timeit
from copy import deepcopy
from dataclasses import asdict

import torch
import torch._utils
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from tqdm.auto import tqdm

from cs336_systems.config import BenchmarkConfig, describe_time_summary, parse_arguments


def setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()


def broadcast_model_parameters(rank, model, debug):
    """Broadcasts model parameters from rank 0 to all other ranks."""
    if debug:
        print(
            f"Rank {rank}: Before broadcast, "
            f"model has weights: {model.state_dict()['token_embeddings.weight'][0, :10]}"
        )

    for param in model.parameters():
        # The broadcast happens in-place for the receiving tensors.
        dist.broadcast(param.data, src=0)

    # Add a barrier to ensure all ranks have received the parameters before proceeding
    dist.barrier()

    if debug:
        print(
            f"Rank {rank}: After broadcast, "
            f"model has weights: {model.state_dict()['token_embeddings.weight'][0, :10]}"
        )


def all_reduce_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            # Sum gradients from all processes
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, async_op=False)
            # Average the gradients by dividing by the world size
            # param.grad.data /= world_size


def all_reduce_gradients_flat(model):
    # use torch._utils._flatten_dense_tensors and torch._utils._unflatten_dense_tensors
    flat_gradient_list = [p.grad.data for p in model.parameters() if p.grad is not None]
    flat_gradients = torch._utils._flatten_dense_tensors(flat_gradient_list)
    dist.all_reduce(flat_gradients, op=dist.ReduceOp.AVG, async_op=False)
    unflattened_gradients = torch._utils._unflatten_dense_tensors(
        flat_gradients, flat_gradient_list
    )
    for p, g in zip(flat_gradient_list, unflattened_gradients):
        p.grad.data.copy_(g)


def ddp_per_worker(rank: int, world_size: int, config: BenchmarkConfig):
    device = setup_process_group(rank, world_size, "nccl" if torch.cuda.is_available() else "gloo")

    torch.manual_seed(rank)
    model_config = config.model_config
    model = BasicsTransformerLM(**asdict(model_config)).to(device)
    print(f"rank {rank}: device: {device}, model: {model}")

    broadcast_model_parameters(rank, model, config.debug)
    non_parallel_model = deepcopy(model)

    global_batch_size = config.batch_size
    local_batch_size = global_batch_size // world_size
    print(
        f"Rank {rank}: Global batch size: {global_batch_size}, Local batch size: {local_batch_size}"
    )
    local_input = torch.zeros(
        local_batch_size, model_config.context_length, device=device, dtype=torch.int64
    )

    if rank == 0:
        global_batch_list = [
            torch.randint(
                low=0,
                high=model_config.vocab_size,
                size=(local_batch_size, model_config.context_length),
                device=device,
            )
            for _ in range(world_size)
        ]
    else:
        global_batch_list = None
    dist.scatter(local_input, global_batch_list, src=0)

    dist.barrier()
    if config.debug:
        print(f"Rank {rank}: Received input: {local_input}")

    if rank == 0:
        global_batch_input = torch.cat(global_batch_list, dim=0)

    optimizer = AdamW(model.parameters())
    non_parallel_optimizer = AdamW(non_parallel_model.parameters())

    for non_parallel_model_parameter, model_parameter in zip(
        non_parallel_model.parameters(), model.parameters()
    ):
        assert torch.allclose(non_parallel_model_parameter, model_parameter)

    num_iterations = config.num_trials
    if config.debug and rank == 0:
        for _ in tqdm(
            range(num_iterations), desc=f"Rank {rank} training loop for non-parallel model"
        ):
            non_parallel_optimizer.zero_grad()
            # Run the non-parallel model on all the data and take a gradient step
            non_parallel_outputs = non_parallel_model(global_batch_input)
            non_parallel_loss = cross_entropy(non_parallel_outputs, global_batch_input)
            non_parallel_loss.backward()
            non_parallel_optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    dist.barrier()
    print(f"Rank {rank}: Starting DDP training loop")
    train_step_times = torch.zeros(num_iterations)
    comm_times = torch.zeros(num_iterations)

    for step in tqdm(range(num_iterations), desc=f"Rank {rank} training loop"):
        start_time = timeit.default_timer()
        optimizer.zero_grad()

        predictions = model(local_input)
        loss = cross_entropy(predictions, local_input)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_communication_time = timeit.default_timer()
        all_reduce_gradients(model)
        end_communication_time = timeit.default_timer()
        comm_times[step] = end_communication_time - start_communication_time

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        train_step_times[step] = end_time - start_time

        if config.debug and rank == 0:
            # veryfying that the named parameters of the non-parallel model match the DDP model
            # print what parameters are different
            for (non_parallel_model_name, non_parallel_model_parameter), (
                model_name,
                model_parameter,
            ) in zip(non_parallel_model.named_parameters(), model.named_parameters()):
                assert torch.allclose(
                    non_parallel_model_parameter, model_parameter, rtol=1e-3, atol=1e-5
                ), (
                    f"Mismatch in {non_parallel_model_name} and {model_name}"
                    f" with values {non_parallel_model_parameter} and {model_parameter}"
                )

    dist.barrier()
    print(f"Rank {rank}: Finished DDP training loop")

    train_times_list = [torch.zeros(world_size) for _ in range(world_size)]
    dist.all_gather_object(train_times_list, train_step_times)
    train_times = torch.cat(train_times_list, dim=0)
    comm_times_list = [torch.zeros(world_size) for _ in range(world_size)]
    dist.all_gather_object(comm_times_list, comm_times)
    comm_times = torch.cat(comm_times_list, dim=0)
    if rank == 0:
        print(f"Rank {rank}: DDP training step times: {train_times}")
        print(f"Rank {rank}: DDP communication times: {comm_times}")
        describe_time_summary(train_times, "DDP training step", config.time_unit)
        describe_time_summary(comm_times, "DDP communication step", config.time_unit)

    cleanup_process_group()


DEBUG_CONFIG = CONFIG = BenchmarkConfig(
    model_config_name="small",
    batch_size=16,
    num_warmups=0,
    num_trials=1,
    time_unit="ms",
    debug=True,
)

CONFIG = BenchmarkConfig(
    model_config_name="medium",
    batch_size=16,
    num_warmups=0,
    num_trials=10,
    time_unit="ms",
    debug=False,
)

if __name__ == "__main__":
    config = parse_arguments(default=CONFIG)

    world_size = 2
    mp.spawn(
        fn=ddp_per_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )
