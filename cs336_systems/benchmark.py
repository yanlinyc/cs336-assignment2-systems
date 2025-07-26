import timeit
from dataclasses import asdict

import cs336_basics.model
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from tqdm.auto import tqdm

from cs336_systems.config import BenchmarkConfig, describe_time_summary, parse_arguments

cs336_basics.model.scaled_dot_product_attention = (
    cs336_basics.model.annotated_scaled_dot_product_attention
)


@nvtx.range("Benchmarking Transformer Model")
def benchmark(config: BenchmarkConfig) -> None:
    """Benchmark transformer model performance."""
    model = BasicsTransformerLM(**asdict(config.model_config)).to(config.device)
    optimizer = AdamW(model.parameters())

    input_tensor = torch.randint(
        low=0,
        high=config.model_config.vocab_size,
        size=(config.batch_size, config.model_config.context_length),
    ).to(config.device)

    # Warmup phase
    with nvtx.range("Warmup Phase"):
        for _ in tqdm(range(config.num_warmups), desc="Warming up"):
            predictions = model(input_tensor)
            loss = cross_entropy(predictions, input_tensor)

            if config.include_backward:
                loss.backward()
                optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmarking phase
    forward_times: list[float] = []
    backward_times: list[float] = []

    nvtx.range_push("Benchmarking Phase")
    for step in tqdm(range(config.num_trials), desc="Benchmarking"):
        nvtx.range_push(f"Step_{step}")
        start_time = timeit.default_timer()

        # Forward pass
        nvtx.range_push("Forward Pass")
        optimizer.zero_grad()
        predictions = model(input_tensor)
        loss = cross_entropy(predictions, input_tensor)
        nvtx.range_pop()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end_time = timeit.default_timer()
        forward_times.append(forward_end_time - start_time)

        # Backward pass
        if config.include_backward:
            with nvtx.range("Backward Pass"):
                loss.backward()

            with nvtx.range("Optimizer Step"):
                optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end_time = timeit.default_timer()
            backward_times.append(backward_end_time - forward_end_time)
        nvtx.range_pop()
    nvtx.range_pop()

    describe_time_summary(forward_times, "Forward pass", config.time_unit)
    if config.include_backward:
        describe_time_summary(backward_times, "Backward pass", config.time_unit)


if __name__ == "__main__":
    benchmark(parse_arguments())
