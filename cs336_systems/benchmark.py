import timeit
from dataclasses import dataclass

import torch
import tyro
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from tqdm.auto import tqdm


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    batch_size: int = 4
    rope_theta: float = 10000.0

    context_length: int = 256
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8


PREDEFINED_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xlarge": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass
class BenchmarkConfig:
    model_config_name: str = "small"
    model_config: ModelConfig | None = None
    num_warmups: int = 5
    num_trials: int = 10
    device: str = "cuda"
    include_backward: bool = True
    time_unit: str = "seconds"  # Options: "seconds", "ms"


def benchmark(
    model_config: ModelConfig,
    num_warmups: int = 5,
    num_trials: int = 10,
    device: str = "cuda",
    include_backward: bool = True,
    time_unit: str = "seconds",
) -> None:
    """Benchmark transformer model performance."""
    model = BasicsTransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
    ).to(device)
    optimizer = AdamW(model.parameters())

    input_tensor = torch.randint(
        low=0,
        high=model_config.vocab_size,
        size=(model_config.batch_size, model_config.context_length),
    ).to(device)

    # Warmup phase
    for _ in tqdm(range(num_warmups), desc="Warming up"):
        predictions = model(input_tensor)
        loss = cross_entropy(predictions, input_tensor)

        if include_backward:
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmarking phase
    forward_times: list[float] = []
    backward_times: list[float] = []

    for _ in tqdm(range(num_trials), desc="Benchmarking"):
        start_time = timeit.default_timer()

        # Forward pass
        optimizer.zero_grad()
        predictions = model(input_tensor)
        loss = cross_entropy(predictions, input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end_time = timeit.default_timer()
        forward_times.append(forward_end_time - start_time)

        # Backward pass
        if include_backward:
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end_time = timeit.default_timer()
            backward_times.append(backward_end_time - forward_end_time)

    def _time_summary(times: list[float], label: str) -> tuple[float, float, float, float]:
        """Calculate and display timing statistics."""
        # Convert times based on the specified unit
        unit_multiplier = 1000 if time_unit == "ms" else 1
        unit_label = "ms" if time_unit == "ms" else "seconds"

        converted_times = [t * unit_multiplier for t in times]

        mean_time = sum(converted_times) / len(converted_times)
        std_time = (
            sum((t - mean_time) ** 2 for t in converted_times) / len(converted_times)
        ) ** 0.5
        min_time = min(converted_times)
        max_time = max(converted_times)

        print(f"{label} - Mean time ± Std: {mean_time:.6f} ± {std_time:.6f} {unit_label}")
        print(
            f"{label} - Min time: {min_time:.6f} {unit_label}, Max time: {max_time:.6f} {unit_label}"
        )
        return mean_time, std_time, min_time, max_time

    _time_summary(forward_times, "Forward pass")
    if include_backward:
        _time_summary(backward_times, "Backward pass")


def main():
    """Main entry point for the benchmark script."""
    config = tyro.cli(BenchmarkConfig)

    if config.model_config_name in PREDEFINED_CONFIGS:
        config.model_config = PREDEFINED_CONFIGS[config.model_config_name]

    assert config.model_config is not None, "Model configuration must be provided."

    print(f"Benchmark configuration: {config}")

    benchmark(
        model_config=config.model_config,
        num_warmups=config.num_warmups,
        num_trials=config.num_trials,
        device=config.device,
        include_backward=config.include_backward,
        time_unit=config.time_unit,
    )


if __name__ == "__main__":
    main()
